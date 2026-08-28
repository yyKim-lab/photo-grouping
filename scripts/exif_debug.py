#!/usr/bin/env python3
"""Diagnostic: does the Picker API's processing-size fetch (=w-h) carry ANY
EXIF at all, and does a much larger size or no size param behave
differently? Written after location_spike.py came back 0/181 geotagged —
need to know whether that's "these photos have no GPS" or "this fetch mode
strips EXIF entirely," before trusting either conclusion.

Pick ONE photo you're sure has location data (e.g. a recent phone photo
taken outdoors with Location Services on for the camera app).

Usage:
    .venv/bin/python scripts/exif_debug.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import ExifTags, Image
from io import BytesIO

from photo_grouping import google_auth, picker_client  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_SECRET_PATH = REPO_ROOT / "secrets" / "client_secret.json"
TOKEN_CACHE_PATH = REPO_ROOT / "secrets" / "token.json"


def dump_exif(label: str, data: bytes) -> None:
    print(f"--- {label} ({len(data):,} bytes) ---")
    img = Image.open(BytesIO(data))
    e = img.getexif()
    if not e:
        print("  No top-level EXIF at all.")
    else:
        for tag_id, value in e.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            print(f"  {tag_name} ({tag_id}): {value!r}")
    gps_ifd = e.get_ifd(ExifTags.IFD.GPSInfo) if e else {}
    if gps_ifd:
        print("  GPS IFD:")
        for tag_id, value in gps_ifd.items():
            tag_name = ExifTags.GPSTAGS.get(tag_id, tag_id)
            print(f"    {tag_name} ({tag_id}): {value!r}")
    else:
        print("  No GPS IFD.")
    print()


def main() -> None:
    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)

    session = picker_client.create_session(access_token)
    print(f"Open this URL and pick 2-3 MORE photos you know have GPS data (different dates/locations ideally):\n\n  {session['pickerUri']}\n")
    session = picker_client.poll_until_ready(access_token, session)
    items = picker_client.list_media_items(access_token, session["id"])

    if not items:
        print("No items picked.")
        return

    print(f"\n{len(items)} item(s) picked. Checking the processing-size fetch (=w1024-h1024) on each:\n")

    results = []
    for item in items:
        base_url = item["mediaFile"]["baseUrl"]
        filename = item["mediaFile"].get("filename", item["id"])
        try:
            data = picker_client._fetch_base_url(access_token, f"{base_url}=w1024-h1024")
        except Exception as e:  # noqa: BLE001
            print(f"- {filename}: FETCH FAILED ({e})")
            continue

        img = Image.open(BytesIO(data))
        e = img.getexif()
        gps_ifd = e.get_ifd(ExifTags.IFD.GPSInfo) if e else {}
        has_gps = bool(gps_ifd)
        results.append((filename, has_gps))
        print(f"- {filename}: {'GPS PRESENT' if has_gps else 'no GPS'}  (top-level EXIF tags: {len(e)})")

    picker_client.delete_session(access_token, session["id"])

    n_with_gps = sum(1 for _, has_gps in results if has_gps)
    print(f"\nSummary: {n_with_gps}/{len(results)} photo(s) had GPS on the processing fetch.")


if __name__ == "__main__":
    main()
