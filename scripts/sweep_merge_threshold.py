"""Sweep visual_merge_threshold on existing tracklet data (CPU only, no GPU/API needed)."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from videorag._entity_anchor.tracker_v2 import link_visual_tracklets
from videorag._unified_graph.registry import EntityRegistry


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_sweep(tracklets_path: Path, video_id: str, thresholds: list[float]):
    tracklets = load_json(tracklets_path)
    print(f"Loaded {len(tracklets)} tracklets from {tracklets_path.name}")
    print(f"Entity types: {Counter(t['entity_type'] for t in tracklets)}")
    print()

    header = f"{'Threshold':>10} | {'Entities':>8} | {'ANIMAL':>7} | {'PERSON':>7} | {'OTHER':>6} | {'Singletons':>10} | {'Max grp':>7} | {'Mean grp':>8}"
    print(header)
    print("-" * len(header))

    for thr in thresholds:
        registry = EntityRegistry()
        config = {"visual_merge_threshold": thr}
        entities = link_visual_tracklets(tracklets, registry, video_id, config)

        type_counts = Counter(e["entity_type"] for e in entities)
        group_sizes = [len(e["tracklets"]) for e in entities]
        singletons = sum(1 for s in group_sizes if s == 1)
        max_grp = max(group_sizes) if group_sizes else 0
        mean_grp = sum(group_sizes) / len(group_sizes) if group_sizes else 0

        animal = type_counts.get("animal", 0)
        person = type_counts.get("person", 0)
        other = len(entities) - animal - person

        print(
            f"{thr:>10.2f} | {len(entities):>8} | {animal:>7} | {person:>7} | {other:>6} "
            f"| {singletons:>10} | {max_grp:>7} | {mean_grp:>8.1f}"
        )

    print()
    print("Singletons = entities with only 1 tracklet (no merge happened)")
    print("Lower threshold → more merging → fewer entities")
    print("Watch for: entities should decrease smoothly, not jump (jump = over-merging)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep visual merge thresholds")
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Path to video workdir containing tracking_base/tracklets.json",
    )
    parser.add_argument(
        "--video-id",
        default="lyd5Q77qHKA",
        help="Video ID for registry (default: lyd5Q77qHKA)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        help="Threshold values to test",
    )
    args = parser.parse_args()

    tracklets_path = args.workdir / "tracking_base" / "tracklets.json"
    if not tracklets_path.exists():
        print(f"ERROR: {tracklets_path} not found")
        sys.exit(1)

    run_sweep(tracklets_path, args.video_id, sorted(args.thresholds))
