"""§4.6 Autobio — drafts a diary-style narrative for one day from its
photos, named people/places, event labels, and any descriptions the user
already added, plus a combined narrative across a date range built from
those daily entries. Includes the per-segment correction flow
(regenerate_segment) — tap a segment, fix a photo's metadata, regenerate
just that segment rather than the whole entry.

Deferred, per explicit scoping decision (see README): the unlabeled-nudge
on/off *setting* — the nudge itself is computed and stored
(has_unlabeled_flag) so the UI can show it; there's no settings page to
turn it off yet.

The LLM is asked to respond with structured JSON (segments tagged with
their source photo ids) rather than free-form prose — exactly what makes
regenerate_segment possible without re-deriving that structure from
scratch.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date as date_cls, timedelta
from typing import Callable

from . import llm, repository

SYSTEM_PROMPT = """\
You help someone write a first-person diary entry about their day, \
based on a list of their photos. Write like a real diary entry — a \
flowing narrative, not a dry inventory of photos ("Photo 1 shows...", \
"Photo 2 shows..."). Group photos that clearly belong to the same \
moment into one segment rather than one segment per photo.

Report facts, don't invent feelings. State who was there, where, and \
what happened, in the order the photos suggest. Do not add or guess at \
emotions, moods, or reactions that aren't directly evidenced (a photo's \
own note, an event name) — no "felt great", "was a comforting time", \
"seemed happy" unless that's explicitly what a note says. If nothing \
notable happened, just say that plainly rather than dressing it up.

Be terse. One short sentence per segment is normal — two only if the \
segment genuinely spans distinct moments. This is a quick record of the \
day, not a piece of creative writing: no scene-setting, no scenery or \
mood description, no restating the same fact two different ways. Don't \
pad a segment just because a photo has little to say about it.

If a photo has an event label attached (e.g. "원필 생일"), treat that as \
the actual occasion — say what the event was, don't just describe people \
being together with no stated reason. Don't invent people, places, or \
events beyond what's given.

Write in whichever language the given names/places/notes/events are \
mostly in — for this app that's very often Korean.

Respond with ONLY a JSON object, no markdown code fences, no other text \
before or after it, in exactly this shape:
{"segments": [{"text": "...", "source_photo_ids": [1, 2]}, ...]}

Every id in source_photo_ids must be one of the photo ids given to you. \
Every photo id given to you should be referenced by at least one \
segment. Segments should appear in chronological order.\
"""


@dataclass
class Segment:
    text: str
    source_photo_ids: list[int]
    edited: bool = False

    def to_dict(self) -> dict:
        return {"text": self.text, "source_photo_ids": self.source_photo_ids, "edited": self.edited}


class NoPhotosForDate(ValueError):
    """Raised rather than calling the LLM with nothing to describe — an
    empty day isn't a narrative-generation failure, it's just an empty
    day, and the caller should say that plainly instead of spending an
    API call to find out."""


# Plain English names for the LLM instruction below — not shown to the
# user (repository.LANGUAGES has the user-facing labels, with native
# script, for the settings dropdown).
LANGUAGE_NAMES = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "uk": "Ukrainian",
    "es": "Spanish",
    "fr": "French",
}


def build_prompt(photos: list[dict], language: str = "ko") -> str:
    lines = [f"Photos from this day, {len(photos)} total, in chronological order:", ""]
    for photo in photos:
        time = (photo.get("taken_at") or "")[11:16] or "unknown time"
        bits = [f"[id {photo['photo_id']}] {time}"]
        if photo.get("people"):
            bits.append("with " + ", ".join(photo["people"]))
        if photo.get("places"):
            bits.append("at " + ", ".join(photo["places"]))
        if photo.get("events"):
            bits.append("event: " + ", ".join(photo["events"]))
        if photo.get("description"):
            bits.append(f'noted: "{photo["description"]}"')
        lines.append(" — ".join(bits))
    lines.append("")
    # Explicit and separate from SYSTEM_PROMPT's softer "whichever language
    # the given data is mostly in" default — that alone produced entries
    # that flipped language day to day depending on what was in the photo
    # metadata. A per-call instruction here (from the narrative_language
    # setting) overrides it with one stable choice.
    lines.append(
        f"Write the entire entry in {LANGUAGE_NAMES.get(language, language)} — regardless of "
        "what language the names, places, notes, or event labels above happen to be in."
    )
    return "\n".join(lines)


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_llm_response(raw_text: str, valid_photo_ids: set[int]) -> list[Segment]:
    # Despite the system prompt asking for bare JSON, models sometimes
    # wrap it in a markdown fence anyway — strip that before parsing
    # rather than failing on it.
    cleaned = _JSON_FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model response wasn't valid JSON: {e}\n\nRaw response:\n{raw_text}") from e

    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"Model response had no usable segments.\n\nRaw response:\n{raw_text}")

    segments = []
    for item in raw_segments:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        # A hallucinated id pointing at a photo that isn't actually in
        # this day would corrupt a later per-segment correction flow
        # (which resolves a segment's photos by id) — drop anything not
        # in the set we actually gave the model, rather than trusting it.
        ids = [pid for pid in (item.get("source_photo_ids") or []) if pid in valid_photo_ids]
        segments.append(Segment(text=text, source_photo_ids=ids))

    if not segments:
        raise ValueError(f"Model response had no non-empty segments.\n\nRaw response:\n{raw_text}")
    return segments


def generate_daily_entry(
    conn: sqlite3.Connection,
    date: str,
    *,
    complete: Callable[..., str] = llm.complete,
    language: str = "ko",
) -> dict:
    """Gathers the day's photos, drafts a narrative via the LLM, and
    stores it (replacing any prior draft for this date, without touching
    an existing edit — see repository.save_autobio_draft). Returns the
    stored entry dict.

    `complete` is injectable (defaults to llm.complete) purely so tests
    can substitute a fake without a live API key, the same pattern
    face_embeddings/picker_client use elsewhere in this codebase.

    `language` is a code from repository.LANGUAGES (e.g. "ko") — callers
    should read the user's narrative_language setting and pass it through
    rather than relying on this default.
    """
    photos = repository.photos_for_date(conn, date)
    if not photos:
        raise NoPhotosForDate(f"No photos found for {date}.")

    prompt = build_prompt(photos, language=language)
    raw_response = complete(prompt, system=SYSTEM_PROMPT)
    valid_ids = {p["photo_id"] for p in photos}
    segments = _parse_llm_response(raw_response, valid_ids)

    draft_text = "\n\n".join(s.text for s in segments)
    has_unlabeled = repository.count_unlabeled_for_date(conn, date) > 0

    with conn:
        repository.save_autobio_draft(
            conn,
            date=date,
            segments=[s.to_dict() for s in segments],
            draft_text=draft_text,
            has_unlabeled=has_unlabeled,
        )
    return repository.get_autobio_entry(conn, date)


# ---------------------------------------------------------------------
# §4.6's per-segment correction flow: tap a segment, step through its
# source photos to fix metadata (via the existing §4.4 photo-edit page —
# see autobio_entry.html), then regenerate just that one segment rather
# than the whole entry.
# ---------------------------------------------------------------------


class NoSuchSegment(ValueError):
    """The requested segment index doesn't exist for that date's entry."""


def regenerate_segment(
    conn: sqlite3.Connection,
    date: str,
    index: int,
    *,
    complete: Callable[..., str] = llm.complete,
    language: str = "ko",
) -> dict:
    """Re-drafts one segment from a prompt scoped to just its own source
    photos — re-read fresh from the DB, since a metadata correction is
    usually *why* someone wants to regenerate a segment: the whole point
    is picking up a name/place/event they just fixed. Replaces that
    segment's text (not marked as a user edit — it's a fresh draft, same
    as a full regenerate isn't), reassembles final_text, and returns the
    updated entry.
    """
    entry = repository.get_autobio_entry(conn, date)
    if entry is None:
        raise NoPhotosForDate(f"No autobio entry for {date}.")
    segments = entry["segments"]
    if not (0 <= index < len(segments)):
        raise NoSuchSegment(f"No segment {index} for {date} (has {len(segments)}).")

    photo_ids = set(segments[index]["source_photo_ids"])
    if not photo_ids:
        raise ValueError("This segment has no source photos to regenerate from.")

    photos = [p for p in repository.photos_for_date(conn, date) if p["photo_id"] in photo_ids]
    if not photos:
        raise ValueError("None of this segment's source photos exist anymore.")

    prompt = (
        build_prompt(photos, language=language)
        + "\n\nWrite about just these photos as a single moment. Respond in the same "
          "JSON shape as always, but with exactly one segment covering all of them together."
    )
    raw_response = complete(prompt, system=SYSTEM_PROMPT)
    new_segments = _parse_llm_response(raw_response, {p["photo_id"] for p in photos})
    combined_text = " ".join(s.text for s in new_segments)

    with conn:
        repository.set_autobio_segment_text(conn, date, index, combined_text, edited=False)
    return repository.get_autobio_entry(conn, date)


# ---------------------------------------------------------------------
# Combined narrative (§4.6 "Combined narrative") — a date range summary
# built from already-generated/edited daily entries, never re-derived
# from raw photo metadata, so per-day corrections carry through.
# ---------------------------------------------------------------------

COMBINE_SYSTEM_PROMPT = """\
You combine several days' diary entries into one cohesive narrative \
covering the whole period, in the same factual, first-person voice as \
the entries themselves — see the rules those entries were written under: \
report what happened, don't add feelings or moods that aren't already \
stated in them. Preserve what actually happened on each day — don't \
invent anything not present in the given entries, and don't add \
emotional color that wasn't already there — but smooth the transitions \
between days rather than just concatenating them, and avoid repeating \
the same phrasing day after day.

Be terse — a day should usually collapse to one short sentence, same as \
the entries you're combining. Don't expand on what a day's entry already \
said just because you're covering more ground now; combining days means \
shorter per-day mentions, not longer ones. Respond with ONLY the combined \
narrative text: no preamble, no headers, no JSON.\
"""


class NoEntriesForRange(ValueError):
    """No day in the range had any photos to draft from — same reasoning
    as NoPhotosForDate, just over a range instead of a single day."""


def _dates_in_range(start_date: str, end_date: str) -> list[str]:
    start = date_cls.fromisoformat(start_date)
    end = date_cls.fromisoformat(end_date)
    if end < start:
        raise ValueError("End date must be on or after the start date.")
    dates = []
    current = start
    while current <= end:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


def generate_combined_narrative(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    *,
    complete: Callable[..., str] = llm.complete,
    language: str = "ko",
) -> dict:
    """Generates any daily entry missing within the range that actually
    has photos (reusing generate_daily_entry — and, importantly, *not*
    touching a day's entry that already exists, so an edit already made
    to it is exactly what gets combined), skips days with no photos at
    all, then asks the LLM to weave the days' final_text together into
    one narrative. Returns the stored summary dict.

    Unlike a daily entry, a summary has no draft/final split (see
    repository.save_autobio_summary's docstring) — regenerating always
    overwrites the stored text.
    """
    dates = _dates_in_range(start_date, end_date)

    entries = []
    for day in dates:
        entry = repository.get_autobio_entry(conn, day)
        if entry is None:
            if not repository.photos_for_date(conn, day):
                continue  # an empty day within the range is fine, just skipped
            entry = generate_daily_entry(conn, day, complete=complete, language=language)
        entries.append(entry)

    if not entries:
        raise NoEntriesForRange(f"No photos found between {start_date} and {end_date}.")

    daily_texts = "\n\n".join(f"[{e['date']}]\n{e['final_text']}" for e in entries)
    daily_texts += (
        f"\n\nWrite the combined narrative in {LANGUAGE_NAMES.get(language, language)} — "
        "regardless of what language the entries above happen to be in."
    )
    combined_text = complete(daily_texts, system=COMBINE_SYSTEM_PROMPT).strip()

    with conn:
        repository.save_autobio_summary(
            conn,
            start_date=start_date,
            end_date=end_date,
            source_entry_ids=[e["id"] for e in entries],
            text=combined_text,
        )
    return repository.get_autobio_summary(conn, start_date, end_date)
