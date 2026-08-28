"""Bulk seed import (§4.5): face-detect a screenshot of Google Photos'
"People & Pets" grid, guess each face's name from the label text sitting
below its thumbnail, and hand both to the caller for the user to confirm or
correct — never trusted blindly, per the spec's own warning, since OCR
misreads freely (see README "Place-name suggestions" for a measured
example) and non-Latin scripts, emoji-adjacent text, or a partially
obscured label all degrade it further.

No grid-layout parsing: rather than assuming a fixed column count or cell
size, each detected face is matched to whichever OCR text sits directly
below it (roughly centered, not too far down) — spatial proximity, not
grid math. The spec explicitly allows this ("no need to explicitly parse
grid layout").
"""

from __future__ import annotations

from io import BytesIO
from typing import TypedDict

from PIL import Image

from . import face_embeddings, ocr


class SeedCandidate(TypedDict):
    bounding_box: dict
    embedding: list[float]
    guessed_name: str  # "" when no label was found near this face


def _face_pixel_box(bounding_box: dict, width: int, height: int) -> tuple[float, float, float, float]:
    left = bounding_box["x"] * width
    top = bounding_box["y"] * height
    right = left + bounding_box["width"] * width
    bottom = top + bounding_box["height"] * height
    return left, top, right, bottom


def detect_seed_candidates(image_bytes: bytes) -> list[SeedCandidate]:
    """One entry per detected face, in detection order. guessed_name is
    empty when nothing plausible sat below that face — the caller must
    still show the crop for the user to type a name manually."""
    width, height = Image.open(BytesIO(image_bytes)).size

    faces = face_embeddings.detect_faces_in_bytes(image_bytes)
    text_items = ocr.read_text_with_positions(image_bytes)

    candidates: list[SeedCandidate] = []
    for bounding_box, embedding in faces:
        left, top, right, bottom = _face_pixel_box(bounding_box, width, height)
        face_width = right - left

        best_text = ""
        best_distance = None
        for item in text_items:
            text_left, text_top, text_right, _text_bottom = item["box"]
            text_center_x = (text_left + text_right) / 2

            # Roughly under the thumbnail horizontally...
            if not (left - face_width * 0.5 <= text_center_x <= right + face_width * 0.5):
                continue
            # ...and below it, not overlapping or above (a small negative
            # tolerance allows a label that touches the thumbnail's edge).
            distance = text_top - bottom
            if distance < -face_width * 0.1:
                continue
            # Too far below to plausibly be this face's own label rather
            # than the next row's caption.
            if distance > face_width * 1.5:
                continue

            if best_distance is None or distance < best_distance:
                best_text, best_distance = item["text"], distance

        candidates.append(
            {"bounding_box": bounding_box, "embedding": embedding, "guessed_name": best_text}
        )
    return candidates
