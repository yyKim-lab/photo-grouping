#!/usr/bin/env python3
"""Location clustering + reverse geocoding spike — spec §6 build-order step 6.

"Simpler than faces — implement once GPS extraction is confirmed working
from real EXIF data."

Needs a *fresh* Picker session, not the photos already in data/originals/:
GPS EXIF only survives on the processing-size fetch (baseUrl + w/h), since
Google strips it specifically on the '=d' full-quality download used to
save originals (§1, §3 step 5) — confirmed empirically, see README. So
this re-authenticates (reusing the cached refresh token — no new consent
needed) and asks you to pick a fresh batch, ideally including some photos
taken outdoors with location services on.

Does NOT touch the database, and does NOT save any bytes to disk — the
processing fetch is used in memory for EXIF extraction only, per §3 step 2.

Usage:
    .venv/bin/python scripts/location_spike.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import exif, geocoding, google_auth, location_clustering, picker_client  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_SECRET_PATH = REPO_ROOT / "secrets" / "client_secret.json"
TOKEN_CACHE_PATH = REPO_ROOT / "secrets" / "token.json"


def main() -> None:
    if not CLIENT_SECRET_PATH.exists():
        print(f"Missing {CLIENT_SECRET_PATH} — see README.md 'Google OAuth setup'.")
        raise SystemExit(1)

    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)

    session = picker_client.create_session(access_token)
    print(f"Session created: {session['id']}")
    print(f"Open this URL and select a few photos, ideally taken outdoors with location on:\n\n  {session['pickerUri']}\n")

    def on_wait(elapsed: float, _session: dict) -> None:
        print(f"  ...still waiting ({elapsed:.0f}s elapsed)")

    session = picker_client.poll_until_ready(access_token, session, on_wait=on_wait)
    items = picker_client.list_media_items(access_token, session["id"])
    print(f"\n{len(items)} item(s) picked.\n")

    photo_names: list[str] = []
    clusters: list[location_clustering.LocationClusterCandidate] = []
    no_gps_count = 0

    for index, item in enumerate(items):
        media_file = item["mediaFile"]
        base_url = media_file["baseUrl"]
        filename = media_file.get("filename") or item["id"]
        photo_names.append(filename)

        try:
            processing_bytes = picker_client.fetch_processing_bytes(access_token, base_url)
        except Exception as e:  # noqa: BLE001 - per-item guard, matches picker_spike.py's pattern
            print(f"- {filename}: FAILED to fetch ({e})")
            continue

        taken_at = exif.extract_taken_at(processing_bytes)
        gps = exif.extract_gps(processing_bytes)

        if gps is None:
            no_gps_count += 1
            print(f"- {filename}: taken_at={taken_at}, no GPS EXIF (§3 step 7 — manual assignment only)")
            continue

        lat, lng = gps
        cluster = location_clustering.assign_or_create(lat, lng, clusters, photo_index=index)
        print(f"- {filename}: taken_at={taken_at}, gps=({lat:.5f}, {lng:.5f}) -> cluster #{clusters.index(cluster)}")

    picker_client.delete_session(access_token, session["id"])

    print(f"\n{len(clusters)} location cluster(s) from {len(items) - no_gps_count} geotagged photo(s) "
          f"({no_gps_count} had no GPS).\n")

    for i, cluster in enumerate(clusters):
        names = [photo_names[idx] for idx in cluster.photo_indices]
        print(f"Cluster #{i}: {cluster.count} photo(s) — {names}")
        print(f"  centroid: ({cluster.centroid_lat:.5f}, {cluster.centroid_lng:.5f})")
        try:
            place_name = geocoding.reverse_geocode(cluster.centroid_lat, cluster.centroid_lng)
            print(f"  reverse-geocoded hint: {place_name}")
        except Exception as e:  # noqa: BLE001 - geocoding failure shouldn't kill the summary
            print(f"  reverse-geocoding FAILED: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
