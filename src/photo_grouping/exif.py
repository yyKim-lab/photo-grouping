"""EXIF extraction — taken_at + GPS lat/lng — from photo bytes (§3 step 3).

Must run against the *processing* fetch, never the saved original: Google
strips GPS EXIF specifically on '=d' downloads (§1, §3 step 5, §4.6b), so
by the time an original is saved locally its location data is already
gone. GPS has to be read from the processing-size fetch, before that
stripping applies — which is also why ingestion fetches processing bytes
before/alongside the original rather than deriving everything from one
fetch.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Optional

from PIL import ExifTags, Image


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    degrees, minutes, seconds = (float(x) for x in dms)
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_taken_at(image_bytes: bytes) -> Optional[datetime]:
    """Prefers DateTimeOriginal (when the shutter fired) over DateTime
    (which can be a file-modified timestamp) — matches EXIF convention."""
    exif = Image.open(BytesIO(image_bytes)).getexif()
    raw = exif.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def extract_gps(image_bytes: bytes) -> Optional[tuple[float, float]]:
    """Returns (lat, lng) in decimal degrees, or None if the photo has no
    GPS EXIF (§3 step 7: photo falls into the 'no location data' bucket,
    manual assignment only)."""
    exif = Image.open(BytesIO(image_bytes)).getexif()
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if not gps_ifd:
        return None

    lat = gps_ifd.get(ExifTags.GPS.GPSLatitude)
    lat_ref = gps_ifd.get(ExifTags.GPS.GPSLatitudeRef)
    lng = gps_ifd.get(ExifTags.GPS.GPSLongitude)
    lng_ref = gps_ifd.get(ExifTags.GPS.GPSLongitudeRef)
    if not (lat and lat_ref and lng and lng_ref):
        return None

    return _dms_to_decimal(lat, lat_ref), _dms_to_decimal(lng, lng_ref)
