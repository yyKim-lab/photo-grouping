"""Minimal client for the Google Photos Picker API.

Endpoints, query params, and behavior confirmed against Google's docs as of
2026-08:
  https://developers.google.com/photos/picker/reference/rest
  https://developers.google.com/photos/picker/guides/sessions
  https://developers.google.com/photos/picker/guides/media-items

Per the spec's own warning (§1, §6 step 1): re-verify these against the
live docs if requests start failing — Google's scope/endpoint names can
shift over time and this was last checked at implementation time, not
continuously.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

API_BASE = "https://photospicker.googleapis.com/v1"


def _request(method: str, url: str, access_token: str, body: Optional[dict] = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {access_token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def create_session(access_token: str) -> dict:
    """POST /v1/sessions — returns {id, pickerUri, pollingConfig, mediaItemsSet}."""
    return _request("POST", f"{API_BASE}/sessions", access_token, body={})


def get_session(access_token: str, session_id: str) -> dict:
    """GET /v1/sessions/{id} — poll this until mediaItemsSet is true."""
    return _request("GET", f"{API_BASE}/sessions/{session_id}", access_token)


def delete_session(access_token: str, session_id: str) -> None:
    """DELETE /v1/sessions/{id} — Google's docs call this "a best practice",
    not a requirement, and it always runs *after* the items it cleans up
    have already been fetched and ingested. So a failure here must never
    propagate: a real crash observed in practice was a 404 on this exact
    call (the session had likely already expired/been cleaned up
    server-side) taking down the whole response — after ingestion had
    already committed real photos to the database. The user saw a 500 and
    no results page for an import that had, in fact, succeeded.
    """
    try:
        req = urllib.request.Request(f"{API_BASE}/sessions/{session_id}", method="DELETE")
        req.add_header("Authorization", f"Bearer {access_token}")
        urllib.request.urlopen(req).close()
    except urllib.error.URLError:
        pass


def poll_until_ready(
    access_token: str,
    session: dict,
    on_wait: Optional[Callable[[float, dict], None]] = None,
) -> dict:
    """Blocks, honoring the session's own pollInterval/timeoutIn (§3 step 1),
    until the user finishes picking (mediaItemsSet=true). Raises TimeoutError
    if the session's own timeoutIn elapses first."""
    polling = session["pollingConfig"]
    interval = float(str(polling["pollInterval"]).rstrip("s"))
    timeout = float(str(polling["timeoutIn"]).rstrip("s"))
    session_id = session["id"]

    elapsed = 0.0
    while elapsed < timeout:
        time.sleep(interval)
        elapsed += interval
        session = get_session(access_token, session_id)
        if on_wait:
            on_wait(elapsed, session)
        if session.get("mediaItemsSet"):
            return session
    raise TimeoutError(f"Picker session {session_id} timed out after {timeout}s without a selection.")


def list_media_items(access_token: str, session_id: str) -> list[dict]:
    """GET /v1/mediaItems?sessionId=... — handles pagination via nextPageToken."""
    items: list[dict] = []
    page_token = None
    while True:
        params = {"sessionId": session_id, "pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        url = f"{API_BASE}/mediaItems?{urllib.parse.urlencode(params)}"
        page = _request("GET", url, access_token)
        items.extend(page.get("mediaItems", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            return items


def fetch_processing_bytes(access_token: str, base_url: str, max_dim: int = 1024) -> bytes:
    """Small fetch for face detection / EXIF (§3 step 2). Never persisted —
    the caller must discard this after processing."""
    return _fetch_base_url(access_token, f"{base_url}=w{max_dim}-h{max_dim}")


def fetch_original_bytes(access_token: str, base_url: str) -> bytes:
    """Full-quality fetch for permanent storage (§3 step 5, §5). Google
    strips GPS EXIF on '=d' downloads specifically — all other EXIF is
    retained, and our own PhotoLocation data comes from the processing
    fetch instead, taken before this stripping applies."""
    return _fetch_base_url(access_token, f"{base_url}=d")


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _fetch_base_url(access_token: str, url: str, max_attempts: int = 4) -> bytes:
    """Transient errors — 5xx (observed: 504 Gateway Timeout) and 429 Too
    Many Requests (observed: bulk-fetching 114 originals back-to-back) —
    show up often enough over a batch of dozens of sequential fetches that
    a lone hiccup shouldn't be fatal. Any other 4xx (e.g. an expired
    baseUrl past the ~60min window) won't succeed on retry, so fail fast.

    Honors a Retry-After header when Google sends one (seconds, the only
    form this API uses); falls back to exponential backoff otherwise."""
    last_error: urllib.error.HTTPError | None = None
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {access_token}")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_STATUS_CODES or attempt == max_attempts:
                raise
            last_error = e
            retry_after = e.headers.get("Retry-After") if e.headers else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt - 1)
            time.sleep(delay)
    raise last_error  # unreachable, satisfies type checkers
