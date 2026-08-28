#!/usr/bin/env python3
"""Tunes face_clustering.DEFAULT_DISTANCE_THRESHOLD for centroid mode.

The eps≈0.31 in README's "Face embedding" section was tuned against batch
DBSCAN, which chains transitively; ingestion uses incremental
nearest-centroid assignment (§3), which doesn't. Measured on the same 136
faces the two disagreed (65 vs 73 clusters), so the threshold needs its own
number.

This drives the real face_clustering.assign() rather than a reimplementation
of it, so the result applies to the actual pipeline.

Scoring reuses the objective signal from the DBSCAN tuning run: a cluster
containing two faces from the *same photo* is provably wrong, since one
photo can't show the same person twice. No eyeballing required to know a
threshold is too loose.

Centroid mode is also order-dependent in a way DBSCAN isn't — a face can
join a cluster created moments earlier, so the arrival order of photos
changes the outcome. --shuffle-trials measures how much, which says whether
a tuned number is robust or an artifact of one particular ingestion order.

Usage:
    .venv/bin/python scripts/face_threshold_tune_centroid.py
    .venv/bin/python scripts/face_threshold_tune_centroid.py --shuffle-trials 20
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import face_clustering, face_embeddings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "anomaly_spike_faces.npz"


def cluster_incrementally(faces, threshold: float) -> list[int]:
    """Replays ingestion's assignment loop over `faces` in order, returning
    a cluster label per face. Mirrors ingestion._ingest_one exactly:
    assign() against the running centroids, else start a new cluster."""
    centroids: list[face_clustering.FaceClusterCentroid] = []
    labels: list[int] = []

    for face in faces:
        embedding = list(face.encoding)
        cluster = face_clustering.assign(embedding, centroids, threshold=threshold)
        if cluster is None:
            cluster = face_clustering.FaceClusterCentroid(
                cluster_id=len(centroids), centroid=embedding
            )
            centroids.append(cluster)
        else:
            cluster.add(embedding)
        labels.append(cluster.cluster_id)

    return labels


def same_photo_violations(faces, labels) -> int:
    """Count of faces sharing a cluster with another face from the same
    photo — provably wrong groupings."""
    seen: dict[tuple[int, str], int] = {}
    for face, label in zip(faces, labels):
        key = (label, face.photo_path.name)
        seen[key] = seen.get(key, 0) + 1
    return sum(count - 1 for count in seen.values() if count > 1)


def summarize(faces, labels) -> dict:
    sizes: dict[int, int] = {}
    for label in labels:
        sizes[label] = sizes.get(label, 0) + 1
    return {
        "clusters": len(sizes),
        "largest": max(sizes.values()) if sizes else 0,
        "singletons": sum(1 for s in sizes.values() if s == 1),
        "violations": same_photo_violations(faces, labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--range", type=float, nargs=3, default=(0.20, 0.60, 0.02), metavar=("START", "STOP", "STEP"))
    parser.add_argument(
        "--shuffle-trials",
        type=int,
        default=0,
        help="Re-run each threshold with N shuffled photo orders to measure order sensitivity.",
    )
    parser.add_argument(
        "--min-face-px",
        type=int,
        default=0,
        help=(
            "Ignore detections narrower than this. Very small detections are "
            "disproportionately false positives (a repeating decorative pattern in the "
            "test set produced four 36px 'faces'), and their embeddings are unreliable "
            "even when genuine — both of which distort the violation count."
        ),
    )
    args = parser.parse_args()

    if not args.cache_path.exists():
        print(f"No embedding cache at {args.cache_path}. Run scripts/gps_anomaly_spike.py first.")
        raise SystemExit(1)

    faces = face_embeddings.load_cache(args.cache_path)
    total = len(faces)
    if args.min_face_px:
        faces = [f for f in faces if (f.location[1] - f.location[3]) >= args.min_face_px]
        print(f"Filtered to faces >= {args.min_face_px}px wide: {len(faces)}/{total} kept.")
    n_photos = len({f.photo_path.name for f in faces})
    print(f"{len(faces)} face(s) across {n_photos} photo(s), in ingestion order.\n")

    start, stop, step = args.range
    header = f"{'thresh':>7}  {'clusters':>8}  {'largest':>7}  {'singletons':>10}  {'violations':>10}"
    if args.shuffle_trials:
        header += f"  {'clusters across ' + str(args.shuffle_trials) + ' shuffles':>32}"
    print(header)

    zero_violation_thresholds = []
    threshold = start
    while threshold <= stop + 1e-9:
        labels = cluster_incrementally(faces, threshold)
        stats = summarize(faces, labels)
        line = (
            f"{threshold:7.2f}  {stats['clusters']:8d}  {stats['largest']:7d}  "
            f"{stats['singletons']:10d}  {stats['violations']:10d}"
        )

        if args.shuffle_trials:
            counts = []
            for trial in range(args.shuffle_trials):
                rng = random.Random(trial)
                # Shuffle whole photos, not individual faces — ingestion
                # always processes a photo's faces together.
                by_photo: dict[str, list] = {}
                for face in faces:
                    by_photo.setdefault(face.photo_path.name, []).append(face)
                order = list(by_photo)
                rng.shuffle(order)
                shuffled = [face for name in order for face in by_photo[name]]
                counts.append(summarize(shuffled, cluster_incrementally(shuffled, threshold))["clusters"])
            line += f"  {min(counts):3d}-{max(counts):<3d} (mean {statistics.mean(counts):5.1f})".rjust(33)

        print(line)
        if stats["violations"] == 0:
            zero_violation_thresholds.append(threshold)
        threshold += step

    if zero_violation_thresholds:
        best = max(zero_violation_thresholds)
        print(
            f"\nLargest threshold with zero same-photo violations: {best:.2f}\n"
            "That is the loosest setting that provably never merges two different people\n"
            "who appear in one photo. Going looser trades correctness for fewer clusters;\n"
            "going tighter only over-splits one person across more clusters, which the\n"
            "merge action (§4.3) can fix — whereas a false merge hides one person inside\n"
            "another's cluster silently."
        )
    else:
        print("\nNo threshold in this range avoided same-photo violations — try a lower --range.")


if __name__ == "__main__":
    main()
