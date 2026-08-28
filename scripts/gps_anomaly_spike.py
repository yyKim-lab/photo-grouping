#!/usr/bin/env python3
"""Runs the GPS-anomaly check against real data: Takeout GPS + real face
clusters over the photos in data/originals/.

Validates gps_anomalies.py end to end, including the "same person appears
in both" constraint that needs actual face clustering to mean anything.
Not DB-wired — this stitches the pieces together directly, the same way
the ingestion pipeline will once step 7 lands.

Usage:
    .venv/bin/python scripts/gps_anomaly_spike.py --takeout-dir path/to/Takeout
    .venv/bin/python scripts/gps_anomaly_spike.py --takeout-dir path/to/Takeout --no-require-shared-face
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import face_embeddings, gps_anomalies, takeout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINALS_DIR = REPO_ROOT / "data" / "originals"
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "anomaly_spike_faces.npz"
TUNED_EPS = 0.31  # empirically tuned, see README "Face embedding"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--takeout-dir", type=Path, required=True)
    parser.add_argument("--originals-dir", type=Path, default=DEFAULT_ORIGINALS_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--eps", type=float, default=TUNED_EPS)
    parser.add_argument("--max-time-delta-seconds", type=float, default=gps_anomalies.DEFAULT_MAX_TIME_DELTA_SECONDS)
    parser.add_argument("--max-speed-kmh", type=float, default=gps_anomalies.DEFAULT_MAX_PLAUSIBLE_SPEED_KMH)
    parser.add_argument(
        "--high-confidence-only",
        action="store_true",
        help="Only report pairs where the same face cluster appears in both photos.",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore the face cache, re-run detection.")
    args = parser.parse_args()

    import numpy as np
    from sklearn.cluster import DBSCAN

    print(f"Scanning Takeout export at {args.takeout_dir}...")
    records = takeout.scan_takeout_export(args.takeout_dir)
    gps_by_filename = {r.filename: r for r in records}
    print(f"  {len(records)} record(s) with GPS.\n")

    # Only bother with photos that actually have GPS — the check is a no-op
    # without it, and face detection is expensive.
    photo_paths = [
        p
        for p in sorted(args.originals_dir.iterdir())
        if p.is_file() and p.name in gps_by_filename
    ]
    if not photo_paths:
        print("No local originals matched the Takeout export. Run scripts/picker_spike.py first.")
        raise SystemExit(1)
    print(f"{len(photo_paths)} local photo(s) have Takeout GPS. Running face detection...\n")

    if args.cache_path.exists() and not args.refresh:
        print(f"Loading cached face embeddings from {args.cache_path}\n")
        faces = face_embeddings.load_cache(args.cache_path)
    else:
        def on_progress(path: Path, n_faces: int) -> None:
            print(f"  {path.name}: {n_faces} face(s)")

        faces = face_embeddings.extract_faces(photo_paths, on_progress=on_progress)
        face_embeddings.save_cache(faces, args.cache_path)
        print(f"\nCached {len(faces)} face(s) to {args.cache_path}\n")

    if faces:
        encodings = np.array([f.encoding for f in faces])
        labels = DBSCAN(eps=args.eps, min_samples=1, metric="euclidean").fit_predict(encodings)
    else:
        labels = []

    faces_by_photo: dict[str, set] = {}
    for face, label in zip(faces, labels):
        faces_by_photo.setdefault(face.photo_path.name, set()).add(int(label))
    n_clusters = len({int(l) for l in labels if l != -1})
    print(f"{len(faces)} face(s) in {n_clusters} cluster(s) across {len(faces_by_photo)} photo(s).\n")

    samples = [
        gps_anomalies.PhotoLocationSample(
            photo_id=path.name,
            taken_at=gps_by_filename[path.name].taken_at,
            lat=gps_by_filename[path.name].lat,
            lng=gps_by_filename[path.name].lng,
            face_cluster_ids=frozenset(faces_by_photo.get(path.name, set())),
        )
        for path in photo_paths
    ]

    anomalies = gps_anomalies.find_gps_anomalies(
        samples,
        max_time_delta_seconds=args.max_time_delta_seconds,
        max_plausible_speed_kmh=args.max_speed_kmh,
        require_shared_face=args.high_confidence_only,
    )

    print(
        f"Checked {len(samples)} geotagged photo(s) for jumps faster than "
        f"{args.max_speed_kmh:g} km/h within {args.max_time_delta_seconds:g}s.\n"
    )
    n_high = sum(1 for a in anomalies if a.confidence is gps_anomalies.Confidence.HIGH)
    print(f"{len(anomalies)} impossible-jump pair(s): {n_high} high-confidence, {len(anomalies) - n_high} low.\n")

    for a in anomalies[:10]:
        speed = "instant" if a.implied_speed_kmh == float("inf") else f"{a.implied_speed_kmh:,.0f} km/h"
        print(f"- [{a.confidence.value:>4}] {a.photo_a_id}  <->  {a.photo_b_id}")
        print(
            f"    {a.distance_m / 1000:.1f} km apart, {a.time_delta_seconds:.0f}s apart "
            f"-> {speed}; shared face cluster(s): {sorted(a.shared_face_cluster_ids) or 'none'}"
        )
    if len(anomalies) > 10:
        print(f"  ... and {len(anomalies) - 10} more pair(s)")

    checklist = gps_anomalies.photos_needing_review(anomalies)
    if checklist:
        print(f"\nManual review checklist ({len(checklist)} photo(s), most-conflicted first):")
        for item in checklist:
            print(f"  [{item.confidence.value:>4}] {item.conflict_count:3d} conflict(s)  {item.photo_id}")


if __name__ == "__main__":
    main()
