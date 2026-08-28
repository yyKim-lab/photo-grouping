#!/usr/bin/env python3
"""Demonstrates backfilling GPS from a Google Takeout export onto photos
already downloaded via scripts/picker_spike.py.

Not DB-wired yet (that's build-order step 7) — this correlates a Takeout
export against the filenames already sitting in data/originals/ and prints
what it finds, so the matching logic can be checked against real Takeout
data before it goes anywhere near the database.

Setup:
    1. Go to https://takeout.google.com, deselect everything except
       "Google Photos", and request an export. This can take anywhere
       from minutes to (for a large library) over a day — Google prepares
       it asynchronously and emails you when it's ready.
    2. Download and extract the export (it may be split across multiple
       .zip files — extract all of them into one directory; this script
       doesn't handle .zip archives directly).
    3. Run this script pointing at that extracted directory.

Usage:
    .venv/bin/python scripts/takeout_backfill_demo.py --takeout-dir /path/to/Takeout
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import geocoding, location_clustering, takeout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINALS_DIR = REPO_ROOT / "data" / "originals"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--takeout-dir", type=Path, required=True, help="Extracted Takeout export directory.")
    parser.add_argument("--originals-dir", type=Path, default=DEFAULT_ORIGINALS_DIR)
    parser.add_argument("--geocode", action="store_true", help="Also reverse-geocode each resulting cluster.")
    parser.add_argument(
        "--newer-than-minutes",
        type=float,
        help=(
            "Only consider originals modified within this many minutes. Useful when "
            "data/originals/ holds photos from more than one Google account (a Takeout "
            "export only covers the account it came from, so photos from other accounts "
            "can never match and would otherwise be counted as misleading 'unmatched' noise)."
        ),
    )
    args = parser.parse_args()

    if not args.takeout_dir.exists():
        print(f"{args.takeout_dir} does not exist.")
        raise SystemExit(1)

    local_paths = [p for p in args.originals_dir.iterdir() if p.is_file()]
    if args.newer_than_minutes is not None:
        cutoff = time.time() - args.newer_than_minutes * 60
        total_before = len(local_paths)
        local_paths = [p for p in local_paths if p.stat().st_mtime >= cutoff]
        print(
            f"Scoped to {len(local_paths)}/{total_before} original(s) modified in the last "
            f"{args.newer_than_minutes:g} minute(s).\n"
        )
    local_filenames = sorted(p.name for p in local_paths)
    if not local_filenames:
        print(f"No files in {args.originals_dir}. Run scripts/picker_spike.py first.")
        raise SystemExit(1)

    print(f"Scanning {args.takeout_dir} for GPS-bearing sidecars (this walks the whole export, may take a bit)...")
    records = takeout.scan_takeout_export(args.takeout_dir)
    print(f"Found {len(records)} media file(s) with usable GPS in the Takeout export.\n")

    matches = takeout.match_records_to_filenames(records, local_filenames)
    print(f"{len(matches)}/{len(local_filenames)} locally-downloaded photo(s) matched to a Takeout GPS record.\n")

    unmatched = [f for f in local_filenames if f not in matches]

    clusters: list[location_clustering.LocationClusterCandidate] = []
    filename_by_index: list[str] = []
    for filename, record in matches.items():
        print(f"- {filename}: ({record.lat:.5f}, {record.lng:.5f})  [via {record.source_json_path.name}]")
        index = len(filename_by_index)
        filename_by_index.append(filename)
        location_clustering.assign_or_create(record.lat, record.lng, clusters, photo_index=index)

    print(f"\n{len(clusters)} location cluster(s) from the matched photos.")
    if unmatched:
        print(f"{len(unmatched)} photo(s) had no Takeout match — falls to 'no location data, manual only' (§3 step 7):")
        for filename in unmatched[:10]:
            print(f"  - {filename}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    if args.geocode:
        print()
        for i, cluster in enumerate(clusters):
            names = [filename_by_index[idx] for idx in cluster.photo_indices]
            place = geocoding.reverse_geocode(cluster.centroid_lat, cluster.centroid_lng)
            print(f"Cluster #{i}: {names} -> {place}")


if __name__ == "__main__":
    main()
