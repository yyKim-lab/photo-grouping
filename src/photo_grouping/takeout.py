"""Google Takeout GPS backfill.

Not part of the original spec — added because the Picker API cannot
provide GPS through any `baseUrl` fetch variant (confirmed empirically;
see README's "Confirmed platform constraint" section), while a Google
Takeout export does preserve it. This module parses an already-extracted
Takeout export and correlates its per-photo JSON sidecars back to
Picker-selected photos by filename, to backfill Photo.gps_lat/gps_lng for
photos the Picker API itself can't supply location for.

Google Takeout's export format is genuinely messy — this is a well-known
problem several community tools exist solely to work around
(GooglePhotosTakeoutHelper, GoogleTakeoutPhotoFixer, etc.):
  - Sidecar naming has changed across export versions:
    `photo.jpg.json` vs `photo.jpg.supplemental-metadata.json`.
  - Google caps the sidecar filename length, truncating into
    ".supplemental-metadata" itself when the combined name is too long.
  - Duplicate photos get a "(n)" suffix that *relocates* between the media
    filename and its sidecar: `IMG(1).jpg` pairs with
    `IMG.jpg.supplemental-metadata(1).json`, not `IMG(1).jpg.json`.

This handles those three documented cases. A media file whose sidecar
doesn't match any of them is skipped, not guessed at — it just falls
through to the existing "no location data, manual assignment only"
bucket (§3 step 7), same as any other ungeotagged photo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".mp4", ".mov"}
_DUP_SUFFIX_RE = re.compile(r"^(.*)\((\d+)\)(\.[^.]+)$")


@dataclass
class TakeoutRecord:
    filename: str  # basename as it appears in the Takeout export
    source_json_path: Path
    lat: float
    lng: float
    taken_at: Optional[datetime]  # UTC, from photoTakenTime — may be absent


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _extract_geo(data: dict) -> Optional[tuple[float, float]]:
    """Prefers geoData; falls back to geoDataExif. Takeout sometimes zeroes
    out one but keeps the other populated — a documented quirk of the
    export, not something we can rely on being consistent."""
    for key in ("geoData", "geoDataExif"):
        geo = data.get(key)
        if not geo:
            continue
        lat, lng = geo.get("latitude"), geo.get("longitude")
        if lat is not None and lng is not None and (lat != 0.0 or lng != 0.0):
            return float(lat), float(lng)
    return None


def _extract_taken_at(data: dict) -> Optional[datetime]:
    ts = data.get("photoTakenTime", {}).get("timestamp")
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _find_sidecar(media_path: Path) -> Optional[Path]:
    """Looks for media_path's JSON sidecar in the same directory, trying
    (in order): the two known exact naming schemes, the "(n)"
    duplicate-suffix relocation, then truncated-prefix matches of either
    exact scheme."""
    directory = media_path.parent
    name = media_path.name

    exact_candidates = [f"{name}.json", f"{name}.supplemental-metadata.json"]
    for candidate_name in exact_candidates:
        candidate = directory / candidate_name
        if candidate.exists():
            return candidate

    dup_match = _DUP_SUFFIX_RE.match(name)
    if dup_match:
        base, dup_n, ext = dup_match.groups()
        relocated = directory / f"{base}{ext}.supplemental-metadata({dup_n}).json"
        if relocated.exists():
            return relocated

    # Truncation: Google cuts the sidecar filename to a length cap, landing
    # mid-way through "supplemental-metadata" — but the ".json" extension
    # itself is always preserved at the end (observed: ".supplemental-
    # metadat.json", ".suppl.json", etc.), so this is a prefix relationship
    # on the *stems* (name minus ".json"), not on the full filenames. Any
    # *.json sibling whose stem is a strict prefix of one of the exact
    # candidates' stems, and at least as long as the original media
    # filename (so it can't be truncating into the base name itself), is a
    # valid match. Prefer the longest such match to minimize false positives.
    best_match: Optional[Path] = None
    try:
        siblings = list(directory.glob("*.json"))
    except OSError:
        return None
    for sibling in siblings:
        if not sibling.name.endswith(".json") or len(sibling.name) < len(name):
            continue
        sibling_stem = sibling.name[: -len(".json")]
        for candidate_name in exact_candidates:
            candidate_stem = candidate_name[: -len(".json")]
            if candidate_stem.startswith(sibling_stem):
                if best_match is None or len(sibling.name) > len(best_match.name):
                    best_match = sibling
    return best_match


def scan_takeout_export(root_dir: Path) -> list[TakeoutRecord]:
    """Walks an already-extracted Takeout export directory (unzip it first
    — this doesn't handle .zip/.tgz archives directly, since large exports
    routinely span several split archives that need combining first
    anyway). Returns one TakeoutRecord per media file that has a matchable
    sidecar with usable GPS. Files without a matchable sidecar, or whose
    sidecar has no GPS, are silently skipped — not an error."""
    records: list[TakeoutRecord] = []
    for media_path in root_dir.rglob("*"):
        if not media_path.is_file() or media_path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        sidecar = _find_sidecar(media_path)
        if sidecar is None:
            continue
        data = _read_json(sidecar)
        if data is None:
            continue
        geo = _extract_geo(data)
        if geo is None:
            continue
        lat, lng = geo
        records.append(
            TakeoutRecord(
                filename=media_path.name,
                source_json_path=sidecar,
                lat=lat,
                lng=lng,
                taken_at=_extract_taken_at(data),
            )
        )
    return records


def match_records_to_filenames(
    records: list[TakeoutRecord], filenames: list[str]
) -> dict[str, TakeoutRecord]:
    """Correlates Takeout records back to a list of Picker-selected
    filenames. When multiple Takeout records share a filename (e.g. the
    same photo exported into two different album folders), the first with
    usable geo data wins — duplicates should agree on location anyway."""
    by_filename: dict[str, TakeoutRecord] = {}
    for record in records:
        by_filename.setdefault(record.filename, record)
    return {name: by_filename[name] for name in filenames if name in by_filename}
