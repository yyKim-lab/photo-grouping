"""Face detection + embedding, and an on-disk cache for tuning runs.

Uses InsightFace's buffalo_l pack (SCRFD detector + ArcFace w600k_r50
recognition, 512-d embeddings) rather than the spec's originally-chosen
face_recognition/dlib.

Why the change — measured, not assumed. On this library's own photos, dlib
ranked *different people* as its most-similar cluster pairs: its closest
pair of all 1596 was an older woman against a younger woman (0.270), and a
young woman against a young man scored 0.314, while genuinely-same-person
pairs sat further apart. That is the documented dlib caveat showing up in
practice — its model is trained predominantly on Western faces and
compresses inter-person distance on Korean ones. The practical cost was
severe fragmentation: at the loosest threshold that never merged two people
from a single photo, dlib produced 57 clusters where ArcFace produced 20
over the same faces, i.e. it split each person roughly 2.5 ways.

ArcFace also has a far wider safe operating range on this data — every
similarity threshold from 0.32 to 0.50 was free of same-photo violations,
versus dlib's knife-edge where 0.33 was safe and 0.34 broke. That matters
more than the headline number: it means the threshold is not fragile.

Embeddings are L2-normalized, so cosine similarity is the natural metric —
see face_clustering.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Detections narrower than this are dropped. Small faces yield unreliable
# embeddings whatever the model, and with dlib's weaker HOG detector this
# also filtered outright false positives — a repeating dancheong painting
# produced four 36px "faces" that clustered together and looked exactly
# like a same-person merge. SCRFD is much less prone to that, but the floor
# is cheap insurance and keeps tiny, low-information faces out of the
# labeling queue.
MIN_FACE_WIDTH_PX = 40

# InsightFace's detector works on a fixed square input; larger sees smaller
# faces at more cost. 640 is the pack's own default.
DETECTION_SIZE = (640, 640)

_analyzer = None


def _get_analyzer():
    """Loads the model once per process. First call downloads ~300MB into
    ~/.insightface/models/ if it isn't cached yet."""
    global _analyzer
    if _analyzer is None:
        from insightface.app import FaceAnalysis

        _analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _analyzer.prepare(ctx_id=-1, det_size=DETECTION_SIZE)
    return _analyzer


@dataclass
class Face:
    photo_path: Path
    location: tuple[int, int, int, int]  # (top, right, bottom, left)
    encoding: np.ndarray  # 512-d, L2-normalized


def list_image_paths(source_dir: Path, limit: Optional[int] = None) -> list[Path]:
    paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit else paths


def _detect(image: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Returns ((top, right, bottom, left), normalized embedding) per face
    above the size floor."""
    faces = []
    for face in _get_analyzer().get(image):
        left, top, right, bottom = (int(v) for v in face.bbox)
        if (right - left) < MIN_FACE_WIDTH_PX:
            continue
        faces.append(((top, right, bottom, left), face.normed_embedding))
    return faces


def _load_bgr(path_or_bytes) -> np.ndarray:
    """InsightFace expects BGR, matching OpenCV convention."""
    from PIL import Image

    source = path_or_bytes if isinstance(path_or_bytes, (str, Path)) else BytesIO(path_or_bytes)
    rgb = np.array(Image.open(source).convert("RGB"))
    return rgb[:, :, ::-1]


def detect_faces_in_bytes(image_bytes: bytes) -> list[tuple[dict, list[float]]]:
    """Detects faces in in-memory image bytes, returning (bounding_box,
    embedding) per face — the shape ingestion.py's detector protocol wants.

    Must be given the *full-size original* bytes: face size in pixels drives
    both the size floor and embedding quality, so detecting at a different
    resolution than the similarity threshold was tuned at silently changes
    clustering behavior.

    Bounding boxes are normalized 0..1 fractions of image width/height, per
    the schema's documented format, so they stay meaningful independent of
    the resolution they were computed at.
    """
    image = _load_bgr(image_bytes)
    height, width = image.shape[:2]

    results = []
    for (top, right, bottom, left), embedding in _detect(image):
        bounding_box = {
            "x": max(0.0, left / width),
            "y": max(0.0, top / height),
            "width": min(1.0, (right - left) / width),
            "height": min(1.0, (bottom - top) / height),
        }
        results.append((bounding_box, [float(v) for v in embedding]))
    return results


# How much padding to add around a user-drawn box before re-detecting
# within it (§4.4 "manually add a face"). A hand-drawn box is rarely a tight
# crop, and the detector needs some context around a face to work at all —
# padding too little risks clipping the face the user was trying to select.
MANUAL_CROP_PADDING = 0.4


def embed_manual_crop(
    image_bytes: bytes, bounding_box: dict
) -> Optional[tuple[dict, list[float]]]:
    """For §4.4's 'manually add a face the detector missed': crops the
    user-drawn region (with padding) out of the full photo and re-runs
    detection on just that crop, rather than trying to embed the drawn box
    directly.

    Two reasons for that indirection rather than a shortcut:
      - ArcFace's recognition model expects a face aligned by the
        detector's own landmarks; feeding it an arbitrary unaligned crop
        would produce a low-quality embedding that wouldn't cluster
        sensibly against auto-detected faces.
      - Re-detecting also gives back a *tight* bounding box instead of
        whatever the user roughly dragged, which is what gets stored and
        shown afterward.

    Returns (bounding_box, embedding) in the *original* image's normalized
    0..1 coordinates, or None if no face was found in the cropped region —
    the caller should tell the user to try drawing a closer box, not treat
    this as an error.

    If multiple faces land in the crop (the drawn box wasn't tight and
    caught a neighbour), the largest is assumed to be the intended one.
    """
    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    left = bounding_box["x"] * width
    top = bounding_box["y"] * height
    right = left + bounding_box["width"] * width
    bottom = top + bounding_box["height"] * height

    pad_x = bounding_box["width"] * width * MANUAL_CROP_PADDING
    pad_y = bounding_box["height"] * height * MANUAL_CROP_PADDING
    crop_left = max(0, int(left - pad_x))
    crop_top = max(0, int(top - pad_y))
    crop_right = min(width, int(right + pad_x))
    crop_bottom = min(height, int(bottom + pad_y))

    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    crop_bytes = BytesIO()
    crop.save(crop_bytes, format="JPEG", quality=95)

    detections = detect_faces_in_bytes(crop_bytes.getvalue())
    if not detections:
        return None

    # Largest by area, in case a neighbouring face was caught too.
    crop_box, embedding = max(detections, key=lambda d: d[0]["width"] * d[0]["height"])

    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    full_box = {
        "x": (crop_left + crop_box["x"] * crop_width) / width,
        "y": (crop_top + crop_box["y"] * crop_height) / height,
        "width": (crop_box["width"] * crop_width) / width,
        "height": (crop_box["height"] * crop_height) / height,
    }
    return full_box, embedding


def extract_faces(image_paths: list[Path], on_progress=None) -> list[Face]:
    """Batch variant over files on disk, for tuning scripts.
    `on_progress(path, n_faces_found)` is called after each photo."""
    faces: list[Face] = []
    for path in image_paths:
        for location, embedding in _detect(_load_bgr(path)):
            faces.append(Face(photo_path=path, location=location, encoding=embedding))
        if on_progress:
            on_progress(path, sum(1 for f in faces if f.photo_path == path))
    return faces


def save_cache(faces: list[Face], cache_path: Path) -> None:
    """Stores extracted faces as .npz — photo paths + bounding boxes as
    parallel string/int arrays, encodings as one 2D float array."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        photo_paths=np.array([str(f.photo_path) for f in faces]),
        locations=np.array([f.location for f in faces], dtype=np.int64),
        encodings=np.array([f.encoding for f in faces], dtype=np.float64),
    )


def load_cache(cache_path: Path) -> list[Face]:
    data = np.load(cache_path)
    return [
        Face(photo_path=Path(str(p)), location=tuple(int(x) for x in loc), encoding=enc)
        for p, loc, enc in zip(data["photo_paths"], data["locations"], data["encodings"])
    ]
