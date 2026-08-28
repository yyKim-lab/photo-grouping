#!/usr/bin/env python3
"""Standalone validation of the Picker flow — spec §6 build-order step 3.

Does NOT touch the database. This only proves the external dependency
chain works end to end: OAuth -> create session -> user picks photos in a
browser -> poll -> list -> fetch processing bytes + full-quality original
-> save original locally. Per the spec, this is "the highest-risk external
dependency — validate first" before building anything on top of it.

One-time setup (you, not this script — see README.md "Google OAuth setup"):
    1. In Google Cloud Console, create a project, enable the "Google Photos
       Picker API" on it, then create an OAuth 2.0 Client ID of type
       "Desktop app". Download its JSON.
    2. Save that JSON as secrets/client_secret.json (already gitignored).

Run:
    python3 scripts/picker_spike.py

This opens a browser for Google sign-in the first time, then prints a
pickerUri — open it and select a few test photos, then return here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import google_auth, picker_client, storage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_SECRET_PATH = REPO_ROOT / "secrets" / "client_secret.json"
TOKEN_CACHE_PATH = REPO_ROOT / "secrets" / "token.json"
ORIGINALS_DIR = REPO_ROOT / "data" / "originals"


def main() -> None:
    if not CLIENT_SECRET_PATH.exists():
        print(f"Missing {CLIENT_SECRET_PATH}")
        print()
        print("Create an OAuth 'Desktop app' client in Google Cloud Console,")
        print("enable the Google Photos Picker API on that project, and save")
        print("the downloaded JSON at the path above. See this script's")
        print("module docstring for the full one-time setup steps.")
        raise SystemExit(1)

    creds = google_auth.load_client_credentials(CLIENT_SECRET_PATH)
    access_token = google_auth.get_access_token(creds, TOKEN_CACHE_PATH)

    session = picker_client.create_session(access_token)
    print(f"Session created: {session['id']}")
    print(f"Open this URL and select a few test photos:\n\n  {session['pickerUri']}\n")

    def on_wait(elapsed: float, _session: dict) -> None:
        print(f"  ...still waiting ({elapsed:.0f}s elapsed)")

    session = picker_client.poll_until_ready(access_token, session, on_wait=on_wait)
    print("Selection complete.\n")

    items = picker_client.list_media_items(access_token, session["id"])
    print(f"{len(items)} item(s) picked.\n")

    adapter = storage.LocalStorageAdapter(ORIGINALS_DIR)

    succeeded = 0
    failed: list[tuple[str, str]] = []  # (filename, error message)

    for item in items:
        media_file = item["mediaFile"]
        base_url = media_file["baseUrl"]
        filename = media_file.get("filename") or f"{item['id']}.jpg"

        # One bad item (e.g. a transient 5xx that outlasts the retries in
        # picker_client._fetch_base_url) shouldn't take the rest of a batch
        # of dozens of photos down with it — log it and keep going.
        try:
            processing_bytes = picker_client.fetch_processing_bytes(access_token, base_url)
            original_bytes = picker_client.fetch_original_bytes(access_token, base_url)
            backend, path = adapter.save_original(original_bytes, filename)
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a per-item guard
            print(f"- {filename}: FAILED ({e})")
            failed.append((filename, str(e)))
            continue

        succeeded += 1
        print(f"- {filename}")
        print(f"    picker_media_id:  {item['id']}")
        print(f"    processing fetch: {len(processing_bytes):,} bytes (discarded, not persisted)")
        print(f"    original fetch:   {len(original_bytes):,} bytes -> saved to {backend}:{path}")

    picker_client.delete_session(access_token, session["id"])
    print(f"\nSession deleted. {succeeded}/{len(items)} succeeded.")
    if failed:
        print(f"{len(failed)} failed:")
        for filename, error in failed:
            print(f"  - {filename}: {error}")


if __name__ == "__main__":
    main()
