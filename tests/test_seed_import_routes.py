"""Bulk seed import web flow (§4.5): upload -> confirm -> save.

Detection/OCR are mocked (covered separately in test_seed_import.py and by
the real-model checks in the session transcript); these tests cover the
route wiring, the base64 round-trip of embeddings through hidden form
fields, and that unchecking a candidate actually excludes it.
"""

import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402


def _fake_png_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 400), color=(50, 50, 50)).save(buf, "PNG")
    buf.seek(0)
    return buf


class SeedImportRouteTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        self.conn = db.connect(self.db_path)
        db.migrate(self.conn)
        web.app.config.update(DB_PATH=self.db_path, TESTING=True)
        self.client = web.app.test_client()

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()


class FormAndDetectTests(SeedImportRouteTestCase):
    def test_form_renders(self):
        response = self.client.get("/seed-import")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"screenshot", response.data)

    def test_detect_without_a_file_is_rejected(self):
        response = self.client.post("/seed-import/detect", data={})
        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.seed_import.detect_seed_candidates")
    def test_detect_with_no_faces_found_is_reported(self, mock_detect):
        mock_detect.return_value = []

        response = self.client.post(
            "/seed-import/detect",
            data={"screenshot": (_fake_png_bytes(), "shot.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No faces found", response.data)

    @patch("photo_grouping.web.seed_import.detect_seed_candidates")
    def test_detect_renders_a_confirmation_row_per_candidate(self, mock_detect):
        mock_detect.return_value = [
            {"bounding_box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
             "embedding": [0.1] * 512, "guessed_name": "엄마"},
            {"bounding_box": {"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2},
             "embedding": [0.2] * 512, "guessed_name": ""},
        ]

        response = self.client.post(
            "/seed-import/detect",
            data={"screenshot": (_fake_png_bytes(), "shot.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"embedding_0", response.data)
        self.assertIn(b"embedding_1", response.data)
        self.assertIn(b'value="\xec\x97\x84\xeb\xa7\x88"', response.data)  # 엄마, prefilled

    @patch("photo_grouping.web.seed_import.detect_seed_candidates")
    def test_unreadable_upload_is_reported_not_a_500(self, mock_detect):
        mock_detect.side_effect = OSError("cannot identify image file")

        response = self.client.post(
            "/seed-import/detect",
            data={"screenshot": (io.BytesIO(b"not an image"), "shot.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)


class SaveTests(SeedImportRouteTestCase):
    def _embed(self, value: float) -> str:
        return base64.b64encode(repository.encode_embedding([value] * 512)).decode()

    def test_saves_only_checked_candidates(self):
        response = self.client.post(
            "/seed-import/save",
            data={
                "count": "2",
                "name_0": "엄마",
                "embedding_0": self._embed(0.1),
                "keep_0": "on",
                "name_1": "안쓸이름",
                "embedding_1": self._embed(0.2),
                # keep_1 omitted — unchecked
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("얼굴 1개 저장됨".encode(), response.data)  # ui_language defaults to Korean
        rows = self.conn.execute("SELECT name FROM seed_face").fetchall()
        self.assertEqual([r["name"] for r in rows], ["엄마"])

    def test_embedding_round_trips_correctly(self):
        original = [i / 1000.0 for i in range(512)]
        encoded = base64.b64encode(repository.encode_embedding(original)).decode()

        self.client.post(
            "/seed-import/save",
            data={"count": "1", "name_0": "테스트", "embedding_0": encoded, "keep_0": "on"},
        )

        stored = self.conn.execute("SELECT embedding FROM seed_face").fetchone()["embedding"]
        decoded = repository.decode_embedding(stored)
        for expected, actual in zip(original, decoded):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_skips_a_checked_candidate_with_no_name(self):
        response = self.client.post(
            "/seed-import/save",
            data={"count": "1", "name_0": "  ", "embedding_0": self._embed(0.1), "keep_0": "on"},
        )

        self.assertIn("얼굴 0개 저장됨".encode(), response.data)  # ui_language defaults to Korean

    def test_source_defaults_to_screenshot_import(self):
        self.client.post(
            "/seed-import/save",
            data={"count": "1", "name_0": "엄마", "embedding_0": self._embed(0.1), "keep_0": "on"},
        )

        row = self.conn.execute("SELECT source FROM seed_face").fetchone()
        self.assertEqual(row["source"], "screenshot_import")

    def test_missing_count_is_rejected(self):
        response = self.client.post("/seed-import/save", data={})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
