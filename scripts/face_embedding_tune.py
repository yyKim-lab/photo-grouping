#!/usr/bin/env python3
"""DBSCAN eps sweep for face clustering — spec §6 build-order step 5 / §7's
"exact eps/min_samples DBSCAN thresholds ... need empirical tuning".

Runs face detection/encoding once (slow), caches it, then tries many eps
values against the cached embeddings (fast) so you can see how cluster
count and max-cluster-size shift before picking a value to inspect visually
with face_embedding_spike.py.

Usage:
    .venv/bin/python scripts/face_embedding_tune.py
    .venv/bin/python scripts/face_embedding_tune.py --limit 30 --eps-range 0.30 0.55 0.02
    .venv/bin/python scripts/face_embedding_tune.py --refresh   # ignore cache, re-run detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import face_embeddings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "originals"
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "face_embeddings_cache.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument(
        "--eps-range", type=float, nargs=3, default=(0.30, 0.55, 0.02), metavar=("START", "STOP", "STEP")
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cache, re-run face detection.")
    args = parser.parse_args()

    import numpy as np
    from sklearn.cluster import DBSCAN

    if args.cache_path.exists() and not args.refresh:
        print(f"Loading cached embeddings from {args.cache_path}")
        faces = face_embeddings.load_cache(args.cache_path)
    else:
        image_paths = face_embeddings.list_image_paths(args.source_dir, limit=args.limit)
        if not image_paths:
            print(f"No images found in {args.source_dir}. Run scripts/picker_spike.py first.")
            raise SystemExit(1)
        print(f"Running face detection on {len(image_paths)} photo(s) (this is the slow part)...\n")

        def on_progress(path: Path, n_faces: int) -> None:
            print(f"- {path.name}: {n_faces} face(s)")

        faces = face_embeddings.extract_faces(image_paths, on_progress=on_progress)
        face_embeddings.save_cache(faces, args.cache_path)
        print(f"\nCached {len(faces)} face(s) to {args.cache_path} for instant re-sweeps.\n")

    if not faces:
        print("No faces to cluster.")
        return

    encodings_matrix = np.array([f.encoding for f in faces])
    n_faces = len(faces)

    photo_paths = [str(f.photo_path) for f in faces]

    def count_same_photo_violations(labels: np.ndarray) -> int:
        """A hard, objective error signal: one photo cannot show the same
        real person twice (barring rare mirror/reflection shots). If a
        cluster contains 2+ faces from the same photo, eps merged two
        different people — no eyeballing required to know that's wrong."""
        violations = 0
        for label in set(labels):
            if label == -1:
                continue
            photos_in_cluster = [photo_paths[i] for i, l in enumerate(labels) if l == label]
            seen: dict[str, int] = {}
            for p in photos_in_cluster:
                seen[p] = seen.get(p, 0) + 1
            violations += sum(count - 1 for count in seen.values() if count > 1)
        return violations

    start, stop, step = args.eps_range
    print(f"{n_faces} face(s) total. Sweeping eps from {start} to {stop} (step {step}), min_samples={args.min_samples}:\n")
    print(f"{'eps':>6}  {'clusters':>8}  {'largest':>7}  {'singletons':>10}  {'same-photo violations':>22}")

    eps = start
    while eps <= stop + 1e-9:
        labels = DBSCAN(eps=eps, min_samples=args.min_samples, metric="euclidean").fit_predict(encodings_matrix)
        sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([])
        n_clusters = len(sizes)
        largest = int(sizes.max()) if len(sizes) else 0
        singletons = int((sizes == 1).sum())
        violations = count_same_photo_violations(labels)
        print(f"{eps:6.2f}  {n_clusters:8d}  {largest:7d}  {singletons:10d}  {violations:22d}")
        eps += step

    print(
        "\n'same-photo violations' is the load-bearing column: it's an "
        "objective proof of over-merging (one photo can't show the same "
        "real person twice), not a judgment call. Pick the largest eps "
        "with zero violations — going smaller than that only over-splits "
        "single people across more clusters without buying back anything, "
        "since split clusters get reconciled later via the merge action "
        "(§4.3) whereas a false merge silently hides one person inside "
        "another's cluster. Once a value looks right, run:\n"
        "  .venv/bin/python scripts/face_embedding_spike.py --eps <value>\n"
        "and look at the cluster_N/ folders to confirm — this check catches "
        "merges, not whether same-person splits are excessive."
    )


if __name__ == "__main__":
    main()
