"""Re-link tracklets with new threshold and overwrite visual_entities + registry.
BACKUP tracking_base/ TRƯỚC KHI CHẠY.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from videorag._entity_anchor.tracker_v2 import link_visual_tracklets
from videorag._unified_graph.registry import EntityRegistry


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Apply new merge threshold")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--video-id", default="lyd5Q77qHKA")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--no-backup", action="store_true", help="Skip backup (DANGEROUS)")
    args = parser.parse_args()

    tracking_dir = args.workdir / "tracking_base"
    tracklets_path = tracking_dir / "tracklets.json"
    entities_path = tracking_dir / "visual_entities.json"
    registry_path = tracking_dir / "registry.json"

    for p in [tracklets_path, entities_path, registry_path]:
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    if not args.no_backup:
        backup_dir = tracking_dir.parent / "tracking_base_backup"
        if backup_dir.exists():
            print(f"Backup already exists at {backup_dir}, skipping backup")
        else:
            print(f"Backing up tracking_base/ -> {backup_dir}")
            shutil.copytree(tracking_dir, backup_dir)

    tracklets = load_json(tracklets_path)
    print(f"Loaded {len(tracklets)} tracklets")

    old_entities = load_json(entities_path)
    print(f"Old: {len(old_entities)} entities (threshold was 0.85)")

    registry = EntityRegistry()
    config = {"visual_merge_threshold": args.threshold}
    new_entities = link_visual_tracklets(tracklets, registry, args.video_id, config)
    print(f"New: {len(new_entities)} entities (threshold = {args.threshold})")
    print(f"Reduction: {len(old_entities)} -> {len(new_entities)} ({len(old_entities) - len(new_entities)} merged away)")

    observations_path = tracking_dir / "observations.json"
    if observations_path.exists():
        observations = load_json(observations_path)
        tracklet_to_entity = {}
        for entity in new_entities:
            for tid in entity["tracklets"]:
                tracklet_to_entity[tid] = entity["entity_id"]
        for obs in observations:
            tid = str(obs.get("tracklet_id", ""))
            if tid in tracklet_to_entity:
                obs["entity_id"] = tracklet_to_entity[tid]
        write_json(observations_path, observations)
        print(f"Updated {len(observations)} observations with new entity IDs")

    write_json(entities_path, new_entities)
    write_json(registry_path, registry.to_dict())
    print(f"Wrote {entities_path.name} and {registry_path.name}")

    alignment_dir = args.workdir / "alignment_caption"
    if alignment_dir.exists():
        segments_dir = alignment_dir / "segments"
        if segments_dir.exists():
            checkpoint = segments_dir / "merge_state.json"
            if checkpoint.exists():
                checkpoint.unlink()
                print(f"Deleted {checkpoint} -- alignment will re-run from batch 0")
            captions_path = segments_dir / "captions.json"
            if captions_path.exists():
                print(f"Kept {captions_path} -- MiniCPM captions reusable")

    print()
    print("NEXT STEPS:")
    print("1. Re-run pipeline with --restart-stage alignment_caption")
    print("2. This will re-do crossmodal merge (GPT-4o-mini) + graph build + embedding")
    print("3. NO GPU detection/tracking needed -- only API calls + CPU")


if __name__ == "__main__":
    main()
