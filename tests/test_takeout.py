import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import takeout  # noqa: E402


def _write_sidecar(path: Path, *, lat=None, lng=None, timestamp=None, geo_key="geoData"):
    payload = {}
    if lat is not None:
        payload[geo_key] = {"latitude": lat, "longitude": lng, "altitude": 0.0}
    if timestamp is not None:
        payload["photoTakenTime"] = {"timestamp": str(timestamp)}
    path.write_text(json.dumps(payload))


class ScanTakeoutExportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_exact_json_suffix_match(self):
        (self.root / "IMG_0001.jpg").write_bytes(b"fake")
        _write_sidecar(self.root / "IMG_0001.jpg.json", lat=37.5, lng=127.0, timestamp=1700000000)

        records = takeout.scan_takeout_export(self.root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].filename, "IMG_0001.jpg")
        self.assertEqual(records[0].lat, 37.5)
        self.assertEqual(records[0].lng, 127.0)
        self.assertEqual(records[0].taken_at, datetime.fromtimestamp(1700000000, tz=timezone.utc))

    def test_supplemental_metadata_suffix_match(self):
        (self.root / "IMG_0002.jpg").write_bytes(b"fake")
        _write_sidecar(self.root / "IMG_0002.jpg.supplemental-metadata.json", lat=1.0, lng=2.0)

        records = takeout.scan_takeout_export(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].filename, "IMG_0002.jpg")

    def test_truncated_supplemental_metadata_suffix(self):
        (self.root / "a_very_long_original_filename_from_camera.jpg").write_bytes(b"fake")
        # Simulates Google cutting the sidecar name mid-suffix.
        truncated_name = "a_very_long_original_filename_from_camera.jpg.supplemental-metadat.json"
        _write_sidecar(self.root / truncated_name, lat=10.0, lng=20.0)

        records = takeout.scan_takeout_export(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].lat, 10.0)

    def test_duplicate_suffix_relocation(self):
        (self.root / "IMG_0003(1).jpg").write_bytes(b"fake")
        _write_sidecar(self.root / "IMG_0003.jpg.supplemental-metadata(1).json", lat=5.0, lng=6.0)

        records = takeout.scan_takeout_export(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].filename, "IMG_0003(1).jpg")
        self.assertEqual(records[0].lat, 5.0)

    def test_falls_back_to_geo_data_exif_when_geo_data_is_zeroed(self):
        (self.root / "IMG_0004.jpg").write_bytes(b"fake")
        payload = {
            "geoData": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
            "geoDataExif": {"latitude": 33.0, "longitude": 44.0, "altitude": 0.0},
        }
        (self.root / "IMG_0004.jpg.json").write_text(json.dumps(payload))

        records = takeout.scan_takeout_export(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].lat, 33.0)

    def test_media_without_matching_sidecar_is_skipped_not_errored(self):
        (self.root / "IMG_0005.jpg").write_bytes(b"fake")
        # No sidecar at all.

        records = takeout.scan_takeout_export(self.root)
        self.assertEqual(records, [])

    def test_sidecar_with_no_gps_is_skipped(self):
        (self.root / "IMG_0006.jpg").write_bytes(b"fake")
        _write_sidecar(self.root / "IMG_0006.jpg.json", timestamp=1700000000)  # no lat/lng

        records = takeout.scan_takeout_export(self.root)
        self.assertEqual(records, [])

    def test_scans_nested_album_directories(self):
        album_dir = self.root / "Google Photos" / "Trip to Busan"
        album_dir.mkdir(parents=True)
        (album_dir / "IMG_0007.jpg").write_bytes(b"fake")
        _write_sidecar(album_dir / "IMG_0007.jpg.json", lat=35.1, lng=129.0)

        records = takeout.scan_takeout_export(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].filename, "IMG_0007.jpg")

    def test_non_media_files_are_ignored(self):
        (self.root / "notes.txt").write_text("hello")
        (self.root / "notes.txt.json").write_text("{}")

        records = takeout.scan_takeout_export(self.root)
        self.assertEqual(records, [])


class MatchRecordsToFilenamesTests(unittest.TestCase):
    def test_matches_by_filename(self):
        records = [
            takeout.TakeoutRecord(filename="a.jpg", source_json_path=Path("a.json"), lat=1.0, lng=2.0, taken_at=None),
            takeout.TakeoutRecord(filename="b.jpg", source_json_path=Path("b.json"), lat=3.0, lng=4.0, taken_at=None),
        ]

        matched = takeout.match_records_to_filenames(records, ["a.jpg", "c.jpg"])

        self.assertEqual(set(matched.keys()), {"a.jpg"})
        self.assertEqual(matched["a.jpg"].lat, 1.0)

    def test_first_record_wins_on_duplicate_filename(self):
        records = [
            takeout.TakeoutRecord(filename="a.jpg", source_json_path=Path("x1.json"), lat=1.0, lng=1.0, taken_at=None),
            takeout.TakeoutRecord(filename="a.jpg", source_json_path=Path("x2.json"), lat=9.0, lng=9.0, taken_at=None),
        ]

        matched = takeout.match_records_to_filenames(records, ["a.jpg"])
        self.assertEqual(matched["a.jpg"].lat, 1.0)


if __name__ == "__main__":
    unittest.main()
