#!/usr/bin/env python3
"""Face detection + embedding + clustering spike — spec §6 build-order step 5.

"Pick a library (start with face_recognition for simplicity), run against
~20-30 real photos, manually inspect whether embeddings cluster sensibly
before committing to a clustering algorithm/threshold."

Not wired to the database — reads photos straight from a local directory
(data/originals/ by default, populated by scripts/picker_spike.py) and
writes cropped face thumbnails into per-cluster folders so you can eyeball
whether DBSCAN actually grouped the same person together. That's the whole
point of the spike: don't trust the eps/min_samples numbers below, look at
the folders.

Reuses the embeddings cache scripts/face_embedding_tune.py writes, if
present, so switching --eps after a tuning sweep is instant instead of
re-running face detection.

Usage:
    .venv/bin/python scripts/face_embedding_spike.py
    .venv/bin/python scripts/face_embedding_spike.py --limit 30 --eps 0.5 --min-samples 1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import face_embeddings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "originals"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "face_spike_output"
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "face_embeddings_cache.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--limit", type=int, default=30, help="Max photos to process (spec suggests ~20-30).")
    parser.add_argument(
        "--eps",
        type=float,
        default=0.31,
        help=(
            "DBSCAN eps. Default is empirically tuned (see "
            "face_embedding_tune.py's 'same-photo violations' check) against "
            "this library's first 30 photos/77 faces — not a universal "
            "constant. Re-run the tune script as more/more-varied photos "
            "get processed; the spec's original placeholder was 0.5, which "
            "measurably over-merged distinct people on this sample."
        ),
    )
    parser.add_argument("--min-samples", type=int, default=1, help="DBSCAN min_samples, per spec §3.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache, re-run face detection.")
    args = parser.parse_args()

    import numpy as np
    from PIL import Image
    from sklearn.cluster import DBSCAN

    if args.cache_path.exists() and not args.refresh:
        print(f"Loading cached embeddings from {args.cache_path}\n")
        faces = face_embeddings.load_cache(args.cache_path)
    else:
        image_paths = face_embeddings.list_image_paths(args.source_dir, limit=args.limit)
        if not image_paths:
            print(f"No images found in {args.source_dir}. Run scripts/picker_spike.py first.")
            raise SystemExit(1)
        print(f"Processing {len(image_paths)} photo(s) from {args.source_dir}...\n")

        def on_progress(path: Path, n_faces: int) -> None:
            print(f"- {path.name}: {n_faces} face(s) detected")

        faces = face_embeddings.extract_faces(image_paths, on_progress=on_progress)
        face_embeddings.save_cache(faces, args.cache_path)

    print(f"\n{len(faces)} total face(s) available.\n")

    if not faces:
        print("No faces detected in this batch — nothing to cluster. Try a different --source-dir or raise --limit.")
        return

    encodings_matrix = np.array([f.encoding for f in faces])
    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="euclidean").fit_predict(encodings_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(
        f"DBSCAN (eps={args.eps}, min_samples={args.min_samples}): "
        f"{n_clusters} cluster(s), {n_noise} noise point(s)\n"
    )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    cluster_sizes: dict[int, int] = {}
    for face, label in zip(faces, labels):
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        cluster_dir = args.output_dir / (f"cluster_{label}" if label != -1 else "noise")
        cluster_dir.mkdir(exist_ok=True)

        top, right, bottom, left = face.location
        img = Image.open(face.photo_path)
        crop = img.crop((left, top, right, bottom))
        crop_name = f"{face.photo_path.stem}_{left}_{top}.jpg"
        crop.convert("RGB").save(cluster_dir / crop_name)

    for label, size in sorted(cluster_sizes.items()):
        name = f"cluster_{label}" if label != -1 else "noise"
        print(f"  {name}: {size} face(s)")

    print(
        f"\nCropped faces written to {args.output_dir}\n"
        "Open each cluster_N/ folder and check: is it really one person? "
        "Any person split across multiple clusters? Any two people merged "
        "into one? That's what decides whether eps needs adjusting."
    )


if __name__ == "__main__":
    main()
