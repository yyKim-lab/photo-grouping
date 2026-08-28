"""Tests for picker_client.py against mocked HTTP responses — no live
Google credentials needed. Verifies request construction (URLs, headers,
pagination, polling loop) matches the documented API behavior; does not
verify Google's actual server responds as documented (that's what
scripts/picker_spike.py is for, run manually against real credentials).
"""

import json
import sys
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import picker_client  # noqa: E402


def _json_response(payload: dict):
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    cm.close = MagicMock()
    return cm


class CreateAndGetSessionTests(unittest.TestCase):
    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_create_session_posts_to_sessions_endpoint(self, mock_urlopen):
        mock_urlopen.return_value = _json_response(
            {"id": "sess-1", "pickerUri": "https://photos.google.com/picker/x", "mediaItemsSet": False}
        )
        result = picker_client.create_session("token-abc")

        self.assertEqual(result["id"], "sess-1")
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://photospicker.googleapis.com/v1/sessions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer token-abc")

    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_get_session_uses_session_id_in_path(self, mock_urlopen):
        mock_urlopen.return_value = _json_response({"id": "sess-1", "mediaItemsSet": True})
        result = picker_client.get_session("token-abc", "sess-1")

        self.assertTrue(result["mediaItemsSet"])
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://photospicker.googleapis.com/v1/sessions/sess-1")


class DeleteSessionTests(unittest.TestCase):
    """Deletion is cleanup, not a load-bearing step — it always runs after
    the items it's cleaning up have already been fetched and ingested, so
    a failure here must never surface to the caller. Regression coverage
    for a real crash: a 404 (session already gone) took down a whole
    /import/continue response after ingestion had already succeeded and
    committed photos to the database."""

    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_deletes_via_the_sessions_endpoint(self, mock_urlopen):
        picker_client.delete_session("token-abc", "sess-1")

        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://photospicker.googleapis.com/v1/sessions/sess-1")
        self.assertEqual(request.get_method(), "DELETE")
        self.assertEqual(request.get_header("Authorization"), "Bearer token-abc")

    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_swallows_404_without_raising(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://example.com", code=404, msg="Not Found", hdrs=MagicMock(), fp=None
        )

        picker_client.delete_session("token-abc", "sess-1")  # must not raise

    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_swallows_other_url_errors_without_raising(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection reset")

        picker_client.delete_session("token-abc", "sess-1")  # must not raise


class PollUntilReadyTests(unittest.TestCase):
    @patch("photo_grouping.picker_client.time.sleep", return_value=None)
    @patch("photo_grouping.picker_client.get_session")
    def test_returns_as_soon_as_media_items_set(self, mock_get_session, _mock_sleep):
        initial = {
            "id": "sess-1",
            "mediaItemsSet": False,
            "pollingConfig": {"pollInterval": "2s", "timeoutIn": "600s"},
        }
        mock_get_session.side_effect = [
            {**initial, "mediaItemsSet": False},
            {**initial, "mediaItemsSet": True},
        ]

        result = picker_client.poll_until_ready("token", initial)

        self.assertTrue(result["mediaItemsSet"])
        self.assertEqual(mock_get_session.call_count, 2)

    @patch("photo_grouping.picker_client.time.sleep", return_value=None)
    @patch("photo_grouping.picker_client.get_session")
    def test_raises_timeout_error_when_session_expires_unpicked(self, mock_get_session, _mock_sleep):
        session = {
            "id": "sess-1",
            "mediaItemsSet": False,
            "pollingConfig": {"pollInterval": "10s", "timeoutIn": "20s"},
        }
        mock_get_session.return_value = {**session, "mediaItemsSet": False}

        with self.assertRaises(TimeoutError):
            picker_client.poll_until_ready("token", session)


class ListMediaItemsTests(unittest.TestCase):
    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_follows_next_page_token_until_exhausted(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _json_response({"mediaItems": [{"id": "a"}], "nextPageToken": "p2"}),
            _json_response({"mediaItems": [{"id": "b"}]}),
        ]

        items = picker_client.list_media_items("token", "sess-1")

        self.assertEqual([i["id"] for i in items], ["a", "b"])
        self.assertEqual(mock_urlopen.call_count, 2)
        second_request = mock_urlopen.call_args_list[1][0][0]
        self.assertIn("pageToken=p2", second_request.full_url)


class FetchBytesTests(unittest.TestCase):
    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_processing_fetch_uses_width_height_params(self, mock_urlopen):
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = b"thumbnail-bytes"
        mock_urlopen.return_value = cm

        data = picker_client.fetch_processing_bytes("token", "https://example.com/base", max_dim=512)

        self.assertEqual(data, b"thumbnail-bytes")
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://example.com/base=w512-h512")
        self.assertEqual(request.get_header("Authorization"), "Bearer token")

    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_original_fetch_uses_d_param(self, mock_urlopen):
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = b"full-quality-bytes"
        mock_urlopen.return_value = cm

        data = picker_client.fetch_original_bytes("token", "https://example.com/base")

        self.assertEqual(data, b"full-quality-bytes")
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://example.com/base=d")


class FetchRetryTests(unittest.TestCase):
    """Covers the retry behavior added after real runs hit a 504 Gateway
    Timeout partway through a batch of 69 photos, and separately a 429 Too
    Many Requests partway through a batch of 114 — both transient, not bugs
    in request construction. See conversation history for the original runs."""

    def _http_error(self, code: int, headers: dict | None = None) -> urllib.error.HTTPError:
        hdrs = MagicMock()
        hdrs.get.side_effect = lambda key, default=None: (headers or {}).get(key, default)
        return urllib.error.HTTPError(url="https://example.com", code=code, msg="err", hdrs=hdrs, fp=None)

    @patch("photo_grouping.picker_client.time.sleep", return_value=None)
    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_retries_on_5xx_and_eventually_succeeds(self, mock_urlopen, _mock_sleep):
        success = MagicMock()
        success.__enter__.return_value.read.return_value = b"bytes"
        mock_urlopen.side_effect = [self._http_error(504), self._http_error(502), success]

        data = picker_client._fetch_base_url("token", "https://example.com/base=d")

        self.assertEqual(data, b"bytes")
        self.assertEqual(mock_urlopen.call_count, 3)

    @patch("photo_grouping.picker_client.time.sleep", return_value=None)
    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_retries_on_429_and_eventually_succeeds(self, mock_urlopen, _mock_sleep):
        success = MagicMock()
        success.__enter__.return_value.read.return_value = b"bytes"
        mock_urlopen.side_effect = [self._http_error(429), success]

        data = picker_client._fetch_base_url("token", "https://example.com/base=d")

        self.assertEqual(data, b"bytes")
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("photo_grouping.picker_client.time.sleep", return_value=None)
    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_honors_retry_after_header_when_present(self, mock_urlopen, mock_sleep):
        success = MagicMock()
        success.__enter__.return_value.read.return_value = b"bytes"
        mock_urlopen.side_effect = [self._http_error(429, headers={"Retry-After": "7"}), success]

        picker_client._fetch_base_url("token", "https://example.com/base=d")

        mock_sleep.assert_called_once_with(7.0)

    @patch("photo_grouping.picker_client.time.sleep", return_value=None)
    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_gives_up_after_max_attempts_on_persistent_5xx(self, mock_urlopen, _mock_sleep):
        mock_urlopen.side_effect = self._http_error(504)

        with self.assertRaises(urllib.error.HTTPError):
            picker_client._fetch_base_url("token", "https://example.com/base=d", max_attempts=3)

        self.assertEqual(mock_urlopen.call_count, 3)

    @patch("photo_grouping.picker_client.time.sleep", return_value=None)
    @patch("photo_grouping.picker_client.urllib.request.urlopen")
    def test_does_not_retry_on_other_4xx(self, mock_urlopen, _mock_sleep):
        mock_urlopen.side_effect = self._http_error(404)

        with self.assertRaises(urllib.error.HTTPError):
            picker_client._fetch_base_url("token", "https://example.com/base=d")

        self.assertEqual(mock_urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
