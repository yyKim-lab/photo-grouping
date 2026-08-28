"""§4.6b — download a photo's original on request.

The common case needs no Google interaction at all: §5's ingestion
already saves a full-quality original locally for every photo (see
ingestion.py), so this just serves that file directly. The Picker
re-pick flow (a fresh, short session scoped to re-selecting that one
photo) is a fallback for the one case it's actually needed: the local
file is missing. Google interaction there is mocked at the same boundary
the import routes use (google_auth/picker_client).
"""

import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402


class DownloadOriginalTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db_path = self.tmp / "test.db"
        self.conn = db.connect(self.db_path)
        db.migrate(self.conn)
        web.app.config.update(DB_PATH=self.db_path, TESTING=True)
        self.client = web.app.test_client()
        self._n = 0
        # Each test starts with a clean slate — this dict is module-level
        # global state shared across the whole process.
        with web._pending_downloads_lock:
            web._pending_downloads.clear()

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()
        with web._pending_downloads_lock:
            web._pending_downloads.clear()

    def _photo(self, missing_file: bool = False) -> int:
        from PIL import Image

        self._n += 1
        name = f"p{self._n}.jpg"
        path = self.tmp / name
        if not missing_file:
            Image.new("RGB", (200, 200), color=(90, 90, 90)).save(path, "JPEG")
        # missing_file=True: insert a row pointing at a path that was
        # never actually written — simulates a moved/deleted original.
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at="2026-04-12T10:00:00",
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )


class StartPageTests(DownloadOriginalTestCase):
    def test_unknown_photo_is_404(self):
        response = self.client.get("/photo/9999/download-original")
        self.assertEqual(response.status_code, 404)

    def test_common_case_downloads_the_local_file_directly_no_google(self):
        photo_id = self._photo()

        response = self.client.get(f"/photo/{photo_id}/download-original")

        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))
        self.assertIn("p1.jpg", response.headers.get("Content-Disposition", ""))
        self.assertTrue(response.data)  # real JPEG bytes came back

    def test_missing_local_file_falls_back_to_the_picker_page(self):
        photo_id = self._photo(missing_file=True)
        with patch("photo_grouping.web.CLIENT_SECRET_PATH") as mock_path:
            mock_path.exists.return_value = True
            response = self.client.get(f"/photo/{photo_id}/download-original")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"p1.jpg", response.data)
        self.assertIn("시작".encode(), response.data)  # ui_language defaults to Korean

    def test_missing_file_shows_setup_instructions_when_no_credentials(self):
        photo_id = self._photo(missing_file=True)
        with patch("photo_grouping.web.CLIENT_SECRET_PATH") as mock_path:
            mock_path.exists.return_value = False
            response = self.client.get(f"/photo/{photo_id}/download-original")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"p1.jpg", response.data)
        self.assertNotIn(b"Start", response.data)

    def test_photo_detail_page_links_here(self):
        photo_id = self._photo()
        response = self.client.get(f"/photo/{photo_id}")
        self.assertIn(f"/photo/{photo_id}/download-original".encode(), response.data)


class SessionStartTests(DownloadOriginalTestCase):
    def test_missing_credentials_is_reported_not_a_crash(self):
        photo_id = self._photo()
        with patch("photo_grouping.web.CLIENT_SECRET_PATH") as mock_path:
            mock_path.exists.return_value = False
            response = self.client.post(f"/photo/{photo_id}/download-original/start")
        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_creates_a_session_and_shows_the_picker_link(self, mock_path, mock_auth, mock_picker):
        photo_id = self._photo()
        mock_path.exists.return_value = True
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.create_session.return_value = {
            "id": "sess-1", "pickerUri": "https://photos.google.com/picker/xyz"
        }

        response = self.client.post(f"/photo/{photo_id}/download-original/start")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"https://photos.google.com/picker/xyz", response.data)
        with web._pending_downloads_lock:
            self.assertEqual(web._pending_downloads[photo_id]["session_id"], "sess-1")


class StatusTests(DownloadOriginalTestCase):
    def test_missing_session_id_is_rejected(self):
        photo_id = self._photo()
        response = self.client.get(f"/photo/{photo_id}/download-original/status")
        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_reports_ready_true(self, mock_auth, mock_picker):
        photo_id = self._photo()
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}

        response = self.client.get(
            f"/photo/{photo_id}/download-original/status", query_string={"session_id": "sess-1"}
        )

        self.assertEqual(response.get_json(), {"ready": True})

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_expired_session_reports_expired_not_a_crash(self, mock_auth, mock_picker):
        photo_id = self._photo()
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.side_effect = urllib.error.HTTPError(
            url="https://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=None
        )

        response = self.client.get(
            f"/photo/{photo_id}/download-original/status", query_string={"session_id": "sess-1"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ready": False, "expired": True})


class FetchTests(DownloadOriginalTestCase):
    def test_missing_session_id_is_rejected(self):
        photo_id = self._photo()
        response = self.client.post(f"/photo/{photo_id}/download-original/fetch")
        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_not_yet_picked_shows_a_retry_prompt(self, mock_auth, mock_picker):
        photo_id = self._photo()
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": False}

        response = self.client.post(
            f"/photo/{photo_id}/download-original/fetch",
            data={"session_id": "sess-1", "picker_uri": "https://x/picker"},
        )

        self.assertEqual(response.status_code, 200)
        # "아직 선택을 마치지 않았습니다" ("haven't finished picking yet") — ui_language
        # defaults to Korean.
        self.assertIn("아직 선택을 마치지 않았습니다".encode(), response.data)
        mock_picker.list_media_items.assert_not_called()

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_happy_path_downloads_the_file(self, mock_auth, mock_picker):
        photo_id = self._photo()
        with web._pending_downloads_lock:
            web._pending_downloads[photo_id] = {"session_id": "sess-1", "picker_uri": "https://x/picker"}
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = [
            {"id": "m1", "mediaFile": {"filename": "original.jpg", "baseUrl": "https://x/a"}},
        ]
        mock_picker.fetch_original_bytes.return_value = b"fake-original-bytes"

        response = self.client.post(
            f"/photo/{photo_id}/download-original/fetch", data={"session_id": "sess-1"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"fake-original-bytes")
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))
        self.assertIn("original.jpg", response.headers.get("Content-Disposition", ""))

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_happy_path_cleans_up_the_session(self, mock_auth, mock_picker):
        photo_id = self._photo()
        with web._pending_downloads_lock:
            web._pending_downloads[photo_id] = {"session_id": "sess-1", "picker_uri": "https://x/picker"}
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = [
            {"id": "m1", "mediaFile": {"filename": "original.jpg", "baseUrl": "https://x/a"}},
        ]
        mock_picker.fetch_original_bytes.return_value = b"bytes"

        self.client.post(f"/photo/{photo_id}/download-original/fetch", data={"session_id": "sess-1"})

        mock_picker.delete_session.assert_called_once()
        with web._pending_downloads_lock:
            self.assertNotIn(photo_id, web._pending_downloads)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_no_items_picked_is_reported_not_a_crash(self, mock_auth, mock_picker):
        photo_id = self._photo()
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = []

        response = self.client.post(
            f"/photo/{photo_id}/download-original/fetch", data={"session_id": "sess-1"}
        )

        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_expired_session_is_reported_not_a_crash(self, mock_auth, mock_picker):
        photo_id = self._photo()
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.side_effect = urllib.error.HTTPError(
            url="https://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=None
        )

        response = self.client.post(
            f"/photo/{photo_id}/download-original/fetch", data={"session_id": "sess-1"}
        )

        self.assertEqual(response.status_code, 410)


if __name__ == "__main__":
    unittest.main()
