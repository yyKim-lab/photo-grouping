"""Reads text off photos, to suggest names for places.

A storefront sign often names a place better than any map does — the map
knows "영산로75번길, 목포시" where the photo plainly says "코롬방제과점".

Uses RapidOCR, which runs the PP-OCR models on onnxruntime — already a
dependency for face recognition, so this adds models (~13MB for Korean)
rather than a second inference runtime.

**Output is a suggestion, never an answer.** Measured on a real storefront,
OCR read "코롬방" at 0.78 confidence but mangled the rest of the same sign
into "코롬빙제고럼" and "코롬방세기". That is a useful prompt for a human who
recognises the place, and a bad thing to write into the database
unattended — matching §4.5's rule for OCR in the seed-import flow ("do not
trust OCR blindly"). Callers get ranked candidates to show the user.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# Below this the text is usually noise rather than a misread.
MIN_CONFIDENCE = 0.5

# Signs carry a lot that isn't a name: opening hours, phone numbers, street
# name plates, "OPEN". Names are short-ish and contain letters.
MAX_CANDIDATE_LENGTH = 20
MIN_CANDIDATE_LENGTH = 2

_MOSTLY_DIGITS = re.compile(r"^[\d\s\-:./]+$")
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

_engine = None


def _get_engine():
    """Loads the model once per process. First call downloads the Korean
    recognition model into the rapidocr package directory."""
    global _engine
    if _engine is None:
        from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

        _engine = RapidOCR(
            params={
                "Rec.lang_type": LangRec.KOREAN,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
            }
        )
    return _engine


def _is_plausible_name(text: str) -> bool:
    text = text.strip()
    if not (MIN_CANDIDATE_LENGTH <= len(text) <= MAX_CANDIDATE_LENGTH):
        return False
    if _MOSTLY_DIGITS.match(text):
        # "1949" on a building, a phone number, opening hours.
        return False
    return bool(_HAS_LETTER.search(text))


def read_text_with_positions(image_bytes: bytes) -> list[dict]:
    """Returns [{text, confidence, box: (left, top, right, bottom)}] in
    pixel coordinates, unranked (source order) — for §4.5's bulk seed
    import, which needs to know *where* each piece of text sits so it can
    match a name label to whichever face thumbnail sits just above it in a
    screenshot grid. read_text_candidates() (below) answers a different
    question — 'what text is probably a name anywhere in this photo' — and
    deliberately throws position away, so it isn't reused here.

    Same plausibility filter as read_text_candidates: a name-shaped string,
    not a timestamp or a stray digit.
    """
    result = _get_engine()(image_bytes)
    if not getattr(result, "txts", None):
        return []

    items = []
    for text, score, box in zip(result.txts, result.scores, result.boxes):
        text = text.strip()
        if score < MIN_CONFIDENCE or not _is_plausible_name(text):
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        items.append(
            {
                "text": text,
                "confidence": round(float(score), 3),
                "box": (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
            }
        )
    return items


def read_text_candidates(image_path: Path, limit: int = 8) -> list[dict]:
    """Returns [{text, confidence}] ranked most-confident first, filtered to
    strings that could plausibly be a place name.

    OCR confidence measures legibility, not "is this a name" — on a real
    storefront photo, "소화전" (fire hydrant), a street-sign fragment, and
    "자동문" (automatic door) all scored higher than "코롬방", the actual
    bakery name, which placed 6th. There's no reliable local heuristic to
    tell a business name from other signage text, so this errs toward
    showing more candidates rather than tightening a confidence cutoff that
    doesn't actually track relevance — the user is scanning these against
    the photo itself, not autocompleting into a text field blind.
    """
    result = _get_engine()(str(image_path))
    if not getattr(result, "txts", None):
        return []

    seen: set[str] = set()
    candidates = []
    for text, score in sorted(zip(result.txts, result.scores), key=lambda pair: -pair[1]):
        text = text.strip()
        if score < MIN_CONFIDENCE or not _is_plausible_name(text) or text in seen:
            continue
        seen.add(text)
        candidates.append({"text": text, "confidence": round(float(score), 3)})
        if len(candidates) >= limit:
            break
    return candidates


def encode_candidates(candidates: list[dict]) -> Optional[str]:
    """Serializes for location_cluster.ocr_name, which is TEXT holding JSON —
    consistent with how the schema stores bounding_box and autobio segments."""
    return json.dumps(candidates, ensure_ascii=False) if candidates else None


def decode_candidates(raw: Optional[str]) -> list[dict]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate a plain string written by an older version.
        return [{"text": raw, "confidence": None}]
    return value if isinstance(value, list) else []
