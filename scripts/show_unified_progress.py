from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _duration(value: Any) -> str:
    if value is None:
        return "-"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _progress_text(row: dict[str, Any]) -> str:
    completed = row.get("completed")
    total = row.get("total")
    unit = str(row.get("unit") or "")
    if completed is None:
        return "-"
    if total is None:
        return f"{completed:g} {unit}".strip()
    total_value = float(total)
    completed_value = float(completed)
    percent = 100.0 * completed_value / total_value if total_value > 0 else 0.0
    return (
        f"{completed_value:g}/{total_value:g} {unit} "
        f"({percent:5.1f}%)"
    ).strip()


def _details_text(row: dict[str, Any]) -> str:
    details = row.get("details") or {}
    if not isinstance(details, dict):
        return ""
    return " ".join(f"{key}={value}" for key, value in details.items())


def _load_rows(root: Path, *, label: str = "") -> list[dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows
    for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        progress_path = video_dir / "progress.json"
        manifest_path = video_dir / "manifest.json"
        if progress_path.exists():
            row = json.loads(progress_path.read_text(encoding="utf-8"))
            if label:
                row["video_id"] = f"{label}/{row.get('video_id', video_dir.name)}"
            rows.append(row)
        elif manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "video_id": f"{label}/{video_dir.name}" if label else video_dir.name,
                    "stage": "not_started",
                    "completed_stages": [
                        name
                        for name, state in manifest.get("stages", {}).items()
                        if state.get("status") == "done"
                    ],
                }
            )
    return rows


def _load_workdir_rows(workdir: Path, *, recursive: bool) -> list[dict[str, Any]]:
    direct = workdir / "pipeline_v2"
    if direct.is_dir() or not recursive:
        return _load_rows(direct)
    rows = []
    for pipeline_root in sorted(workdir.glob("**/pipeline_v2")):
        label = str(pipeline_root.parent.relative_to(workdir))
        rows.extend(_load_rows(pipeline_root, label=label))
    return rows


def _render_table(rows: list[dict[str, Any]]) -> str:
    headers = ("VIDEO", "STAGE", "PROGRESS", "ELAPSED", "ETA", "DETAILS")
    values = [
        (
            str(row.get("video_id", "-")),
            str(row.get("stage", "-")),
            _progress_text(row),
            _duration(row.get("elapsed_seconds")),
            _duration(row.get("eta_seconds")),
            _details_text(row),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        if values
        else len(headers[index])
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in values
    )
    if not values:
        lines.append("No pipeline progress found yet.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Unified V2 ingest progress.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--json", action="store_true", help="Print raw JSON rows.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Aggregate every nested pipeline_v2 directory under --workdir.",
    )
    args = parser.parse_args()
    workdir = Path(args.workdir)

    while True:
        rows = _load_workdir_rows(workdir, recursive=args.recursive)
        if args.watch and not args.json:
            print("\033[2J\033[H", end="")
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            print(_render_table(rows), flush=True)
        if not args.watch:
            break
        time.sleep(max(args.interval, 0.5))


if __name__ == "__main__":
    main()
