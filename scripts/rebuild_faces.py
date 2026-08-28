#!/usr/bin/env python3
"""Re-detects and re-clusters every face, carrying existing names across.

Needed after a change to the embedding model, the detector, or the size
floor: embeddings from different models aren't comparable, so old
face_instance rows can't be mixed with new ones.

Names are preserved rather than discarded. A rebuilt cluster is a different
grouping, but the *faces* are still in the same photos at the same places,
so each new face inherits the name of whichever old face overlapped it in
that photo (by bounding-box IoU). A new cluster then takes the inherited
name its members agree on.

Where members disagree — two people the old model split, that the new one
merged — the cluster is left unlabeled and the conflict is reported rather
than silently picking a winner. Those are exactly the cases worth looking
at by hand.

Photos, locations, and Autobio entries are untouched.

Usage:
    .venv/bin/python scripts/rebuild_faces.py            # dry run
    .venv/bin/python scripts/rebuild_faces.py --yes      # rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, face_clustering, face_embeddings, repository  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "photo_grouping.db"

# Two boxes over the same face rarely align exactly across detectors, so
# this is deliberately permissive; a face has to be clearly the same one.
MIN_IOU_FOR_INHERITANCE = 0.3


def _iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    intersection = ix * iy
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union > 0 else 0.0


def _snapshot_labels(conn) -> dict[int, list[tuple[dict, str]]]:
    """{photo_id: [(bounding_box, name), ...]} for every currently-named
    face, so names can be re-attached after re-detection."""
    snapshot: dict[int, list[tuple[dict, str]]] = {}
    for row in conn.execute(
        """
        SELECT fi.photo_id, fi.bounding_box, fc.name
        FROM face_instance fi
        JOIN face_cluster fc ON fc.id = fi.face_cluster_id
        WHERE fc.status = 'named' AND fc.name IS NOT NULL
        """
    ):
        snapshot.setdefault(row["photo_id"], []).append(
            (json.loads(row["bounding_box"]), row["name"])
        )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--yes", action="store_true", help="Actually rebuild. Without this, only reports.")
    parser.add_argument("--threshold", type=float, default=face_clustering.DEFAULT_SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}.")
        raise SystemExit(1)

    conn = db.connect(args.db)
    db.migrate(conn)

    photos = conn.execute(
        "SELECT id, original_storage_path, original_filename FROM photo ORDER BY id"
    ).fetchall()
    labels = _snapshot_labels(conn)
    named_count = len({n for entries in labels.values() for _, n in entries})
    existing_faces = conn.execute("SELECT COUNT(*) AS c FROM face_instance").fetchone()["c"]

    print(f"{len(photos)} photo(s), {existing_faces} existing face instance(s).")
    print(f"{sum(len(v) for v in labels.values())} labeled face(s) across {named_count} distinct name(s) "
          f"will be carried across by bounding-box overlap.")

    if not args.yes:
        print(f"\nDry run (threshold {args.threshold}). Re-run with --yes to rebuild.")
        return

    print(f"\nRe-detecting at similarity threshold {args.threshold}...\n")

    centroids: list[face_clustering.FaceClusterCentroid] = []
    inherited: dict[object, list[str]] = {}  # cluster_id -> names its faces carried in
    total_faces = 0

    with conn:
        conn.execute("DELETE FROM face_instance")
        conn.execute("DELETE FROM face_cluster")

        for n, photo in enumerate(photos, 1):
            path = Path(photo["original_storage_path"])
            if not path.exists():
                print(f"  [{n}/{len(photos)}] {photo['original_filename']}: file missing, skipped")
                continue

            try:
                detections = face_embeddings.detect_faces_in_bytes(path.read_bytes())
            except Exception as e:  # noqa: BLE001 - one bad photo shouldn't stop the rebuild
                print(f"  [{n}/{len(photos)}] {photo['original_filename']}: FAILED ({e})")
                continue

            old_for_photo = labels.get(photo["id"], [])

            for bounding_box, embedding in detections:
                cluster = face_clustering.assign(embedding, centroids, threshold=args.threshold)
                if cluster is None:
                    cluster_id = repository.insert_face_cluster(
                        conn, representative_photo_id=photo["id"]
                    )
                    cluster = face_clustering.FaceClusterCentroid(
                        cluster_id=cluster_id, centroid=list(embedding)
                    )
                    centroids.append(cluster)
                else:
                    cluster.add(embedding)

                repository.insert_face_instance(
                    conn,
                    photo_id=photo["id"],
                    face_cluster_id=cluster.cluster_id,
                    bounding_box=bounding_box,
                    embedding=embedding,
                )
                total_faces += 1

                best_name, best_iou = None, MIN_IOU_FOR_INHERITANCE
                for old_box, old_name in old_for_photo:
                    overlap = _iou(bounding_box, old_box)
                    if overlap >= best_iou:
                        best_name, best_iou = old_name, overlap
                if best_name:
                    inherited.setdefault(cluster.cluster_id, []).append(best_name)

            if n % 20 == 0:
                print(f"  [{n}/{len(photos)}] {total_faces} faces, {len(centroids)} clusters so far")

        # Re-apply names where a cluster's inherited labels agree.
        restored, conflicted = 0, []
        for cluster_id, names in inherited.items():
            distinct = sorted(set(names))
            if len(distinct) == 1:
                repository.name_cluster(conn, "face", cluster_id, distinct[0])
                restored += 1
            else:
                conflicted.append((cluster_id, distinct, len(names)))

    print(f"\nRebuilt: {total_faces} face(s) in {len(centroids)} cluster(s).")
    print(f"Names restored automatically: {restored}")

    if conflicted:
        print(
            f"\n{len(conflicted)} cluster(s) merged faces you had labeled differently, so they were "
            "left unlabeled for you to review:"
        )
        for cluster_id, distinct, count in conflicted:
            print(f"  cluster #{cluster_id} ({count} labeled faces): {', '.join(distinct)}")
        print("  Open each in the UI — either it is a genuine merge of one person you had")
        print("  split, or the new threshold is too loose for these particular people.")

    for table, count in repository.counts(conn).items():
        print(f"  {count:6d}  {table}")
    conn.close()


if __name__ == "__main__":
    main()
