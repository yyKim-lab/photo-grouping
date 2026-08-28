#!/usr/bin/env python3
"""The real ingestion entry point — §6 build-order step 7.

Unlike the spike scripts this writes to the database, and is safe to
re-run: photos already imported (matched on picker_media_id) are skipped,
and the GPS/location passes only touch rows that still need them.

Subcommands match the passes the pipeline actually has (see ingestion.py
for why GPS can't happen inline with the Picker fetch):

    ingest.py pick                      # Picker session -> photos + faces
    ingest.py backfill-gps --takeout-dir DIR
    ingest.py cluster-locations
    ingest.py review-gps                # implausible-jump checklist

Or run the first three in sequence:

    ingest.py all --takeout-dir DIR

Database defaults to data/photo_grouping.db; --db points elsewhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import (  # noqa: E402
    db,
    face_embeddings,
    google_auth,
    gps_anomalies,
    ingestion,
    picker_client,
    repository,
    storage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_SECRET_PATH = REPO_ROOT / "secrets" / "client_secret.json"
TOKEN_CACHE_PATH = REPO_ROOT / "secrets" / "token.json"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "photo_grouping.db"
DEFAULT_ORIGINALS_DIR = REPO_ROOT / "data" / "originals"


def _open_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(path)
    applied = db.migrate(conn)
    if applied:
        print(f"Applied migration(s): {', '.join(applied)}\n")
    return conn


def _print_counts(conn) -> None:
    print("\nDatabase now holds:")
    for table, count in repository.counts(conn).items():
        print(f"  {count:6d}  {table}")


def cmd_pick(args, conn) -> None:
    if not CLIENT_SECRET_PATH.exists():
        print(f"Missing {CLIENT_SECRET_PATH} — see README.md 'Google OAuth setup'.")
        raise SystemExit(1)

    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)

    session = picker_client.create_session(access_token)
    print(f"Open this URL and select photos:\n\n  {session['pickerUri']}\n")
    session = picker_client.poll_until_ready(
        access_token, session, on_wait=lambda elapsed, _s: print(f"  ...waiting ({elapsed:.0f}s)")
    )
    items = picker_client.list_media_items(access_token, session["id"])
    print(f"\n{len(items)} item(s) picked. Ingesting...\n")

    adapter = storage.get_adapter(args.storage_backend, args.originals_dir)
    detector = None if args.no_faces else face_embeddings.detect_faces_in_bytes

    result = ingestion.ingest_picked_items(
        conn,
        items,
        access_token=access_token,
        picker_client=picker_client,
        storage_adapter=adapter,
        detect_faces=detector,
        on_progress=lambda filename, status: print(f"- {filename}: {status}"),
    )
    picker_client.delete_session(access_token, session["id"])

    print(
        f"\nImported {result.imported}, skipped {result.skipped_already_imported} already-imported, "
        f"{len(result.failed)} failed."
    )
    if detector:
        print(
            f"{result.faces_detected} face(s) detected, "
            f"{result.face_clusters_created} new cluster(s) created."
        )
    for filename, error in result.failed:
        print(f"  FAILED {filename}: {error}")


def cmd_backfill_gps(args, conn) -> None:
    if not args.takeout_dir.exists():
        print(f"{args.takeout_dir} does not exist. See README.md 'Location clustering'.")
        raise SystemExit(1)

    print(f"Scanning {args.takeout_dir}...")
    result = ingestion.backfill_gps_from_takeout(conn, args.takeout_dir)
    print(f"Backfilled GPS for {result.matched} photo(s).")
    if result.unmatched:
        print(
            f"{len(result.unmatched)} photo(s) had no Takeout GPS record — these stay in the "
            "'no location data, manual assignment only' bucket:"
        )
        for filename in result.unmatched[:10]:
            print(f"  - {filename}")
        if len(result.unmatched) > 10:
            print(f"  ... and {len(result.unmatched) - 10} more")


def cmd_cluster_locations(args, conn) -> None:
    created = ingestion.cluster_locations(conn, threshold_m=args.threshold_m)
    print(f"Location clustering done. {created} new cluster(s) created.")


def cmd_suggest_names(args, conn) -> None:
    print("Filling place-name suggestions (geocoding + OCR)...\n")
    found = ingestion.suggest_location_names(
        conn,
        use_geocoding=not args.no_geocode,
        use_ocr=not args.no_ocr,
        on_progress=lambda cid, source, value: print(f"  cluster #{cid:3d} [{source:8s}] {value}"),
    )
    print(f"\nGeocoded names found: {found['geocoded']}   OCR candidates found: {found['ocr']}")


def cmd_review_gps(args, conn) -> None:
    from datetime import datetime

    samples = []
    for row in repository.load_photos_for_anomaly_check(conn):
        taken_at = row["taken_at"]
        if isinstance(taken_at, str):
            try:
                taken_at = datetime.fromisoformat(taken_at.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                continue
        samples.append(
            gps_anomalies.PhotoLocationSample(
                photo_id=row["photo_id"],
                taken_at=taken_at,
                lat=row["lat"],
                lng=row["lng"],
                face_cluster_ids=row["face_cluster_ids"],
            )
        )

    anomalies = gps_anomalies.find_gps_anomalies(
        samples,
        max_time_delta_seconds=args.max_time_delta_seconds,
        max_plausible_speed_kmh=args.max_speed_kmh,
    )
    checklist = gps_anomalies.photos_needing_review(anomalies)

    print(f"Checked {len(samples)} geotagged photo(s): {len(anomalies)} impossible-jump pair(s).\n")
    if not checklist:
        print("No photos need location review.")
        return

    filenames = {
        row["id"]: row["original_filename"]
        for row in conn.execute("SELECT id, original_filename FROM photo")
    }
    print(f"Manual review checklist ({len(checklist)} photo(s), most-conflicted first):")
    for item in checklist:
        print(
            f"  [{item.confidence.value:>4}] {item.conflict_count:3d} conflict(s)  "
            f"#{item.photo_id} {filenames.get(item.photo_id, '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pick = subparsers.add_parser("pick", help="Run a Picker session and ingest the selection.")
    pick.add_argument("--originals-dir", type=Path, default=DEFAULT_ORIGINALS_DIR)
    pick.add_argument("--storage-backend", default="local", choices=["local", "icloud"])
    pick.add_argument("--no-faces", action="store_true", help="Skip face detection (much faster).")

    backfill = subparsers.add_parser("backfill-gps", help="Fill GPS from a Takeout export.")
    backfill.add_argument("--takeout-dir", type=Path, required=True)

    cluster = subparsers.add_parser("cluster-locations", help="Assign photos to location clusters.")
    cluster.add_argument("--threshold-m", type=float, default=150.0)

    suggest = subparsers.add_parser("suggest-names", help="Reverse-geocode and OCR place names.")
    suggest.add_argument("--no-geocode", action="store_true")
    suggest.add_argument("--no-ocr", action="store_true")

    review = subparsers.add_parser("review-gps", help="List photos with implausible location jumps.")
    review.add_argument("--max-time-delta-seconds", type=float, default=gps_anomalies.DEFAULT_MAX_TIME_DELTA_SECONDS)
    review.add_argument("--max-speed-kmh", type=float, default=gps_anomalies.DEFAULT_MAX_PLAUSIBLE_SPEED_KMH)

    run_all = subparsers.add_parser("all", help="pick -> backfill-gps -> cluster-locations.")
    run_all.add_argument("--takeout-dir", type=Path, required=True)
    run_all.add_argument("--originals-dir", type=Path, default=DEFAULT_ORIGINALS_DIR)
    run_all.add_argument("--storage-backend", default="local", choices=["local", "icloud"])
    run_all.add_argument("--no-faces", action="store_true")
    run_all.add_argument("--threshold-m", type=float, default=150.0)

    args = parser.parse_args()
    conn = _open_db(args.db)

    if args.command == "pick":
        cmd_pick(args, conn)
    elif args.command == "backfill-gps":
        cmd_backfill_gps(args, conn)
    elif args.command == "cluster-locations":
        cmd_cluster_locations(args, conn)
    elif args.command == "suggest-names":
        cmd_suggest_names(args, conn)
    elif args.command == "review-gps":
        cmd_review_gps(args, conn)
    elif args.command == "all":
        cmd_pick(args, conn)
        print()
        cmd_backfill_gps(args, conn)
        print()
        cmd_cluster_locations(args, conn)

    if args.command not in ("review-gps", "suggest-names"):
        _print_counts(conn)
    conn.close()


if __name__ == "__main__":
    main()
