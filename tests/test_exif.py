"""Tests for exif.py. Needs Pillow — run with the .venv (which already has
it as a face_recognition dependency), not the bare-stdlib interpreter the
rest of the test suite runs under:

    .venv/bin/python -m unittest tests/test_exif.py -v
"""

import sys
import unittest
from datetime import datetime
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import exif  # noqa: E402


def _make_jpeg_bytes(*, taken_at: str | None = None, gps: tuple | None = None) -> bytes:
    """Builds a tiny in-memory JPEG with the given EXIF fields, so tests
    don't depend on a real photo file existing on disk."""
    from PIL import ExifTags, Image

    img = Image.new("RGB", (10, 10), color="red")
    image_exif = img.getexif()

    if taken_at is not None:
        image_exif[ExifTags.Base.DateTimeOriginal] = taken_at

    if gps is not None:
        lat, lat_ref, lng, lng_ref = gps
        gps_ifd = image_exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps_ifd[ExifTags.GPS.GPSLatitudeRef] = lat_ref
        gps_ifd[ExifTags.GPS.GPSLatitude] = lat
        gps_ifd[ExifTags.GPS.GPSLongitudeRef] = lng_ref
        gps_ifd[ExifTags.GPS.GPSLongitude] = lng

    buf = BytesIO()
    img.save(buf, format="jpeg", exif=image_exif)
    return buf.getvalue()


class ExtractTakenAtTests(unittest.TestCase):
    def test_parses_date_time_original(self):
        data = _make_jpeg_bytes(taken_at="2026:08:24 10:30:00")
        self.assertEqual(exif.extract_taken_at(data), datetime(2026, 8, 24, 10, 30, 0))

    def test_returns_none_when_absent(self):
        data = _make_jpeg_bytes()
        self.assertIsNone(exif.extract_taken_at(data))


class ExtractGpsTests(unittest.TestCase):
    def test_parses_northern_eastern_hemisphere(self):
        # 37°30'00"N 127°00'00"E -> approx Seoul-area latitude/longitude
        data = _make_jpeg_bytes(gps=((37.0, 30.0, 0.0), "N", (127.0, 0.0, 0.0), "E"))
        lat, lng = exif.extract_gps(data)
        self.assertAlmostEqual(lat, 37.5, places=5)
        self.assertAlmostEqual(lng, 127.0, places=5)

    def test_south_and_west_are_negative(self):
        data = _make_jpeg_bytes(gps=((33.0, 51.0, 0.0), "S", (151.0, 12.0, 0.0), "W"))
        lat, lng = exif.extract_gps(data)
        self.assertLess(lat, 0)
        self.assertLess(lng, 0)

    def test_returns_none_when_absent(self):
        data = _make_jpeg_bytes()
        self.assertIsNone(exif.extract_gps(data))


if __name__ == "__main__":
    unittest.main()
