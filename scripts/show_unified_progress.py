from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Unified V2 ingest progress.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    root = Path(args.workdir) / "pipeline_v2"

    import time

    while True:
        rows = []
        if root.exists():
            for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                progress_path = video_dir / "progress.json"
                manifest_path = video_dir / "manifest.json"
                if progress_path.exists():
                    payload = json.loads(progress_path.read_text(encoding="utf-8"))
                    rows.append(payload)
                elif manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "video_id": video_dir.name,
                            "stage": "not_started",
                            "stages": manifest.get("stages", {}),
                        }
                    )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        if not args.watch:
            break
        time.sleep(max(args.interval, 0.5))


if __name__ == "__main__":
    main()

