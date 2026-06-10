from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from tqdm import tqdm


PIPELINE_VERSION = "unified-graph-v3-ablation"
REMOVED_STAGE_NAMES = {
    "modality_profile",
    "deep_processing",
    "speaker_linking",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows: antivirus may briefly lock a newly-written file; retry on PermissionError
    for _attempt in range(6):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if _attempt == 5:
                raise
            time.sleep(0.05 * (2 ** _attempt))


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class StageDefinition:
    name: str
    dependencies: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()


@dataclass
class StageProgress:
    runner: "StageRunner"
    stage: str
    total: int | float | None
    unit: str
    description: str
    completed: int | float = 0
    details: dict[str, Any] = field(default_factory=dict)
    _bar: Any = field(init=False, default=None)

    def __enter__(self) -> "StageProgress":
        self._bar = tqdm(
            total=self.total,
            initial=self.completed,
            desc=self.description,
            unit=self.unit,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        self.runner.update_progress(
            self.stage,
            completed=self.completed,
            total=self.total,
            unit=self.unit,
            details=self.details,
        )
        return self

    def update(self, amount: int | float = 1, **details: Any) -> None:
        self.completed += amount
        self.details.update(details)
        if self._bar is not None:
            self._bar.update(amount)
            if details:
                self._bar.set_postfix(details, refresh=False)
        self.runner.update_progress(
            self.stage,
            completed=self.completed,
            total=self.total,
            unit=self.unit,
            details=self.details,
        )

    def set(self, completed: int | float, **details: Any) -> None:
        delta = completed - self.completed
        self.update(delta, **details)

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._bar is not None:
            self._bar.close()
        self.runner.update_progress(
            self.stage,
            completed=self.completed,
            total=self.total,
            unit=self.unit,
            details=self.details,
            flush=True,
        )


class StageContext:
    def __init__(self, runner: "StageRunner", stage: str):
        self.runner = runner
        self.stage = stage

    @property
    def video_dir(self) -> Path:
        return self.runner.video_dir

    @property
    def stage_dir(self) -> Path:
        path = self.video_dir / self.stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def config(self) -> dict[str, Any]:
        return self.runner.config

    def path(self, name: str) -> Path:
        return self.stage_dir / name

    def checkpoint_path(self, name: str = "checkpoint.json") -> Path:
        return self.path(name)

    def load_checkpoint(self, name: str = "checkpoint.json", default: Any = None) -> Any:
        return read_json(self.checkpoint_path(name), default)

    def save_checkpoint(self, payload: Any, name: str = "checkpoint.json") -> None:
        atomic_write_json(self.checkpoint_path(name), payload)
        self.runner.event("checkpoint", self.stage, {"path": str(self.checkpoint_path(name))})

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.path(name)
        atomic_write_json(path, payload)
        return path

    def read_json(self, name: str, default: Any = None) -> Any:
        return read_json(self.path(name), default)

    def progress(
        self,
        total: int | float | None,
        unit: str = "item",
        description: str | None = None,
        completed: int | float = 0,
        **details: Any,
    ) -> StageProgress:
        return StageProgress(
            runner=self.runner,
            stage=self.stage,
            total=total,
            unit=unit,
            description=description or self.stage,
            completed=completed,
            details=details,
        )

    def report_metrics(self, **metrics: Any) -> None:
        self.runner.update_stage_metrics(self.stage, metrics)

    def log(self, message: str, level: int = logging.INFO, **details: Any) -> None:
        self.runner.logger.log(level, "[%s] %s", self.stage, message)
        self.runner.event("log", self.stage, {"message": message, **details})


class StageRunner:
    def __init__(
        self,
        working_dir: str | Path,
        video_id: str,
        config: dict[str, Any],
        stages: list[StageDefinition],
        *,
        resume: bool = True,
        restart_stage: str | None = None,
        force: bool = False,
    ):
        self.working_dir = Path(working_dir)
        self.video_id = video_id
        self.config = dict(config)
        self.stages = stages
        self.stage_map = {stage.name: stage for stage in stages}
        self.resume = resume
        self.video_dir = self.working_dir / "pipeline_v2" / video_id
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.video_dir / "manifest.json"
        self.progress_path = self.video_dir / "progress.json"
        self.events_path = self.video_dir / "events.jsonl"
        self.report_path = self.video_dir / "run_report.json"
        self.log_path = self.video_dir / "pipeline.log"
        self.logger = self._build_logger()
        self._last_progress_write = 0.0
        self._run_started = time.time()
        self._stage_started_at: dict[str, float] = {}
        self.manifest = self._load_manifest()
        self._prepare_manifest(restart_stage=restart_stage, force=force)

    def _build_logger(self) -> logging.Logger:
        logger_name = f"videorag.pipeline_v2.{self.video_id}.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = True
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        return logger

    def _load_manifest(self) -> dict[str, Any]:
        manifest = read_json(self.manifest_path, {})
        if not isinstance(manifest, dict):
            manifest = {}
        manifest.setdefault("pipeline_version", PIPELINE_VERSION)
        manifest.setdefault("video_id", self.video_id)
        manifest.setdefault("created_at", _utc_now())
        manifest.setdefault("stages", {})
        return manifest

    def _stage_config(self, definition: StageDefinition) -> dict[str, Any]:
        return {key: self.config.get(key) for key in definition.config_keys}

    def _descendants(self, stage_name: str) -> set[str]:
        descendants = {stage_name}
        changed = True
        while changed:
            changed = False
            for definition in self.stages:
                if definition.name in descendants:
                    continue
                if any(dep in descendants for dep in definition.dependencies):
                    descendants.add(definition.name)
                    changed = True
        return descendants

    def _invalidate(
        self,
        stage_names: set[str],
        reason: str,
        *,
        remove_outputs: bool = True,
    ) -> None:
        for stage_name in stage_names:
            stage_dir = self.video_dir / stage_name
            if remove_outputs and stage_dir.exists():
                shutil.rmtree(stage_dir)
            state = self.manifest["stages"].setdefault(stage_name, {})
            state.update(
                {
                    "status": "pending",
                    "invalidated_at": _utc_now(),
                    "invalidation_reason": reason,
                }
            )

    def _prepare_manifest(self, restart_stage: str | None, force: bool) -> None:
        if force or self.manifest.get("pipeline_version") != PIPELINE_VERSION:
            stage_names = {
                definition.name for definition in self.stages
            } | REMOVED_STAGE_NAMES
            for stage_name in stage_names:
                stage_dir = self.video_dir / stage_name
                if stage_dir.exists():
                    shutil.rmtree(stage_dir)
            self.manifest = {
                "pipeline_version": PIPELINE_VERSION,
                "video_id": self.video_id,
                "created_at": _utc_now(),
                "stages": {},
            }

        if not self.resume:
            self._invalidate(
                {definition.name for definition in self.stages},
                "Resume disabled",
            )

        for definition in self.stages:
            stage_config = self._stage_config(definition)
            config_hash = stable_hash(stage_config)
            state = self.manifest["stages"].setdefault(
                definition.name,
                {"status": "pending"},
            )
            previous_hash = state.get("config_hash")
            if previous_hash and previous_hash != config_hash:
                self._invalidate(
                    self._descendants(definition.name),
                    f"Configuration changed for stage {definition.name}",
                )
            state["config_hash"] = config_hash
            state["config"] = stage_config

        if restart_stage:
            if restart_stage not in self.stage_map:
                raise ValueError(f"Unknown restart stage: {restart_stage}")
            self._invalidate(
                self._descendants(restart_stage),
                f"Explicit restart from {restart_stage}",
            )

        self.manifest["pipeline_version"] = PIPELINE_VERSION
        self.manifest["config_hash"] = stable_hash(self.config)
        self.manifest["updated_at"] = _utc_now()
        atomic_write_json(self.manifest_path, self.manifest)

    def event(self, kind: str, stage: str | None, payload: dict[str, Any]) -> None:
        record = {
            "time": _utc_now(),
            "kind": kind,
            "video_id": self.video_id,
            "stage": stage,
            **payload,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def update_progress(
        self,
        stage: str,
        *,
        completed: int | float,
        total: int | float | None,
        unit: str,
        details: dict[str, Any] | None = None,
        flush: bool = False,
    ) -> None:
        now = time.time()
        if not flush and now - self._last_progress_write < 0.5:
            return
        elapsed = max(
            now - self._stage_started_at.get(stage, self._run_started),
            0.0,
        )
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = None
        if total is not None and rate > 0:
            eta = max(float(total) - float(completed), 0.0) / rate
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "video_id": self.video_id,
            "stage": stage,
            "completed": completed,
            "total": total,
            "unit": unit,
            "elapsed_seconds": elapsed,
            "run_elapsed_seconds": max(now - self._run_started, 0.0),
            "eta_seconds": eta,
            "details": details or {},
            "settings": {
                key: self.config.get(key)
                for key in (
                    "asr_model",
                    "entity_tracking_model",
                    "entity_tracking_fps",
                    "caption_backend",
                    "segment_target_seconds",
                    "frame_min",
                    "frame_max",
                )
            },
            "stage_metrics": {
                name: state.get("metrics", {})
                for name, state in self.manifest.get("stages", {}).items()
                if state.get("metrics")
            },
            "completed_stages": [
                name
                for name, state in self.manifest.get("stages", {}).items()
                if state.get("status") == "done"
            ],
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.progress_path, payload)
        state = self.manifest["stages"].setdefault(stage, {})
        state["completed_units"] = completed
        state["total_units"] = total
        state["unit"] = unit
        atomic_write_json(self.manifest_path, self.manifest)
        self._last_progress_write = now

    def update_stage_metrics(self, stage: str, metrics: dict[str, Any]) -> None:
        state = self.manifest["stages"].setdefault(stage, {})
        state.setdefault("metrics", {}).update(metrics)
        atomic_write_json(self.manifest_path, self.manifest)

    def should_run(self, stage_name: str) -> bool:
        state = self.manifest["stages"].get(stage_name, {})
        if not self.resume:
            return True
        return state.get("status") != "done"

    def output(self, stage_name: str, name: str) -> Path:
        return self.video_dir / stage_name / name

    def require_done(self, definition: StageDefinition) -> None:
        missing = [
            dep
            for dep in definition.dependencies
            if self.manifest["stages"].get(dep, {}).get("status") != "done"
        ]
        if missing:
            raise RuntimeError(
                f"Stage {definition.name} requires unfinished stages: {', '.join(missing)}"
            )

    def run_stage(
        self,
        stage_name: str,
        function: Callable[[StageContext], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        definition = self.stage_map[stage_name]
        self.require_done(definition)
        state = self.manifest["stages"].setdefault(stage_name, {})
        if not self.should_run(stage_name):
            self.logger.info("[%s] Resume: already completed, skipping", stage_name)
            self.event("stage_skipped", stage_name, {"reason": "already_done"})
            self.update_progress(
                stage_name,
                completed=state.get("completed_units", 1),
                total=state.get("total_units", 1),
                unit=state.get("unit", "stage"),
                details={"status": "resumed"},
                flush=True,
            )
            return state.get("result")

        started = time.time()
        self._stage_started_at[stage_name] = started
        state.update(
            {
                "status": "running",
                "started_at": _utc_now(),
                "error": None,
            }
        )
        atomic_write_json(self.manifest_path, self.manifest)
        self.event("stage_started", stage_name, {})
        self.logger.info("[%s] Starting", stage_name)
        context = StageContext(self, stage_name)
        try:
            result = function(context) or {}
            elapsed = time.time() - started
            state.update(
                {
                    "status": "done",
                    "finished_at": _utc_now(),
                    "elapsed_seconds": elapsed,
                    "result": result,
                }
            )
            atomic_write_json(self.manifest_path, self.manifest)
            self.update_progress(
                stage_name,
                completed=state.get("completed_units", 1),
                total=state.get("total_units", 1),
                unit=state.get("unit", "stage"),
                details={"status": "done"},
                flush=True,
            )
            self.event(
                "stage_finished",
                stage_name,
                {"elapsed_seconds": elapsed, "result": result},
            )
            self.logger.info("[%s] Done in %.2fs", stage_name, elapsed)
            return result
        except KeyboardInterrupt:
            state.update(
                {
                    "status": "interrupted",
                    "finished_at": _utc_now(),
                    "elapsed_seconds": time.time() - started,
                }
            )
            atomic_write_json(self.manifest_path, self.manifest)
            self.event("stage_interrupted", stage_name, {})
            self.write_run_report(status="interrupted")
            raise
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            state.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "elapsed_seconds": time.time() - started,
                    "error": error,
                }
            )
            atomic_write_json(self.manifest_path, self.manifest)
            self.event("stage_failed", stage_name, error)
            self.logger.exception("[%s] Failed: %s", stage_name, exc)
            self.write_run_report(status="failed")
            raise

    def write_run_report(
        self,
        *,
        status: str = "complete",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = {
            "pipeline_version": PIPELINE_VERSION,
            "video_id": self.video_id,
            "status": status,
            "started_at": datetime.fromtimestamp(
                self._run_started, timezone.utc
            ).isoformat(),
            "finished_at": _utc_now(),
            "elapsed_seconds": time.time() - self._run_started,
            "config_hash": stable_hash(self.config),
            "config": self.config,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "stages": self.manifest.get("stages", {}),
        }
        if extra:
            report.update(extra)
        atomic_write_json(self.report_path, report)
        self.event("run_report", None, {"status": status, "path": str(self.report_path)})
        return report

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    @contextmanager
    def run_scope(self) -> Iterator["StageRunner"]:
        self.event("run_started", None, {"config_hash": stable_hash(self.config)})
        try:
            yield self
        except BaseException:
            raise
        else:
            self.write_run_report(status="complete")
        finally:
            self.close()
