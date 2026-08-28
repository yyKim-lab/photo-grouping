import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import geocoding  # noqa: E402

SEOUL = (37.5665, 126.9780)
TOKYO = (35.6812, 139.7671)
PARIS = (48.8566, 2.3522)


def _json_response(payload: dict):
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return cm


class GeocodingTestCase(unittest.TestCase):
    def setUp(self):
        # Far in the past so the shared rate limiter never stalls a test.
        geocoding._last_request_time = -1000.0
        self._env = patch.dict("os.environ", {}, clear=False)
        self._env.start()
        for key in ("KAKAO_REST_API_KEY", "YAHOO_JP_CLIENT_ID"):
            __import__("os").environ.pop(key, None)

    def tearDown(self):
        self._env.stop()


class KeyResolutionTests(GeocodingTestCase):
    """Mirrors test_llm.py's ApiKeyResolutionTests — same "env var, then
    secrets file" shape, for both regional geocoding keys."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.kakao_path = Path(self._tmp.name) / "kakao_rest_api_key.txt"
        self.yahoo_jp_path = Path(self._tmp.name) / "yahoo_jp_client_id.txt"

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def test_kakao_prefers_environment_variable(self):
        with patch.dict("os.environ", {"KAKAO_REST_API_KEY": "env-key"}):
            self.assertEqual(geocoding.resolve_kakao_key(self.kakao_path), "env-key")

    def test_kakao_falls_back_to_secrets_file(self):
        self.kakao_path.write_text("file-key\n")
        self.assertEqual(geocoding.resolve_kakao_key(self.kakao_path), "file-key")

    def test_kakao_strips_whitespace_and_takes_first_line(self):
        self.kakao_path.write_text("  file-key  \nsome other note\n")
        self.assertEqual(geocoding.resolve_kakao_key(self.kakao_path), "file-key")

    def test_kakao_returns_none_when_nothing_is_configured(self):
        self.assertIsNone(geocoding.resolve_kakao_key(self.kakao_path))

    def test_yahoo_jp_prefers_environment_variable(self):
        with patch.dict("os.environ", {"YAHOO_JP_CLIENT_ID": "env-id"}):
            self.assertEqual(geocoding.resolve_yahoo_jp_client_id(self.yahoo_jp_path), "env-id")

    def test_yahoo_jp_falls_back_to_secrets_file(self):
        self.yahoo_jp_path.write_text("file-id\n")
        self.assertEqual(geocoding.resolve_yahoo_jp_client_id(self.yahoo_jp_path), "file-id")

    def test_yahoo_jp_returns_none_when_nothing_is_configured(self):
        self.assertIsNone(geocoding.resolve_yahoo_jp_client_id(self.yahoo_jp_path))


class RoutingTests(GeocodingTestCase):
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_falls_back_to_nominatim_with_no_keys_configured(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response({"display_name": "Somewhere, Seoul"})

        self.assertEqual(geocoding.reverse_geocode(*SEOUL), "Somewhere, Seoul")

        # Kakao declined without a key, so only Nominatim was called.
        self.assertEqual(mock_urlopen.call_count, 1)
        self.assertIn("nominatim", mock_urlopen.call_args[0][0].full_url)

    @patch.dict("os.environ", {"KAKAO_REST_API_KEY": "test-key"})
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_korean_coordinates_prefer_kakao_poi(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response(
            {"documents": [{"place_name": "니드짐 구파발점"}]}
        )

        self.assertEqual(geocoding.reverse_geocode(*SEOUL), "니드짐 구파발점")

        request = mock_urlopen.call_args[0][0]
        self.assertIn("dapi.kakao.com", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "KakaoAK test-key")

    @patch.dict("os.environ", {"KAKAO_REST_API_KEY": "test-key"})
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_kakao_is_skipped_outside_korea(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response({"display_name": "Paris, France"})

        self.assertEqual(geocoding.reverse_geocode(*PARIS), "Paris, France")

        self.assertNotIn("kakao", mock_urlopen.call_args[0][0].full_url)

    @patch.dict("os.environ", {"YAHOO_JP_CLIENT_ID": "yahoo-id"})
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_japanese_coordinates_use_yahoo(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response(
            {"Feature": [{"Property": {"Address": "東京都千代田区"}}]}
        )

        self.assertEqual(geocoding.reverse_geocode(*TOKYO), "東京都千代田区")
        self.assertIn("yahooapis.jp", mock_urlopen.call_args[0][0].full_url)


class FallthroughTests(GeocodingTestCase):
    @patch.dict("os.environ", {"KAKAO_REST_API_KEY": "test-key"})
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_provider_with_no_result_falls_through(self, mock_urlopen, _sleep):
        # Kakao returns empty for every category and for the address
        # lookup, then Nominatim answers.
        empty = [_json_response({"documents": []}) for _ in range(7)]
        mock_urlopen.side_effect = empty + [_json_response({"display_name": "Fallback place"})]

        self.assertEqual(geocoding.reverse_geocode(*SEOUL), "Fallback place")

    @patch.dict("os.environ", {"KAKAO_REST_API_KEY": "test-key"})
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_provider_error_falls_through_rather_than_raising(self, mock_urlopen, _sleep):
        import urllib.error

        def responses(request, *a, **kw):
            if "kakao" in request.full_url:
                raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", None, None)
            return _json_response({"display_name": "Still works"})

        mock_urlopen.side_effect = responses

        # A bad key shouldn't break geocoding — the hint is advisory.
        self.assertEqual(geocoding.reverse_geocode(*SEOUL), "Still works")

    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_returns_none_when_nothing_has_an_answer(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response({})
        self.assertIsNone(geocoding.reverse_geocode(0.0, 0.0))


class ConfigTests(GeocodingTestCase):
    def test_reports_only_nominatim_without_keys(self):
        self.assertEqual(geocoding.configured_providers(), ["nominatim"])

    @patch.dict("os.environ", {"KAKAO_REST_API_KEY": "k", "YAHOO_JP_CLIENT_ID": "y"})
    def test_reports_configured_providers(self):
        self.assertEqual(geocoding.configured_providers(), ["kakao", "yahoo_japan", "nominatim"])

    def test_user_agent_is_latin_1_encodable(self):
        # HTTP headers are latin-1 encoded; a typographic character raises
        # UnicodeEncodeError inside http.client, which mocking urlopen
        # hides entirely. This regression actually shipped once.
        geocoding.USER_AGENT.encode("latin-1")


class ForwardGeocodingTests(GeocodingTestCase):
    """§4.4: typing a place name to set a photo's location, rather than
    needing to already know its coordinates."""

    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_falls_back_to_nominatim_with_no_keys_configured(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response([{"lat": "48.8584", "lon": "2.2945"}])

        result = geocoding.forward_geocode("Eiffel Tower")

        self.assertEqual(result, (48.8584, 2.2945))
        self.assertEqual(mock_urlopen.call_count, 1)
        self.assertIn("nominatim", mock_urlopen.call_args[0][0].full_url)

    @patch.dict("os.environ", {"KAKAO_REST_API_KEY": "test-key"})
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_kakao_answers_a_korean_place_name(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response(
            {"documents": [{"place_name": "코롬방제과점", "x": "126.9780", "y": "37.5665"}]}
        )

        result = geocoding.forward_geocode("코롬방제과점")

        self.assertEqual(result, (37.5665, 126.978))
        request = mock_urlopen.call_args[0][0]
        self.assertIn("dapi.kakao.com", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "KakaoAK test-key")

    @patch.dict("os.environ", {"YAHOO_JP_CLIENT_ID": "yahoo-id"})
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_yahoo_answers_a_japanese_place_name(self, mock_urlopen, _sleep):
        # YOLP's own coordinate order is "lon,lat" — the opposite of what
        # this function returns; a wrong swap here would silently save a
        # photo's location on the wrong side of the globe.
        mock_urlopen.return_value = _json_response(
            {"Feature": [{"Geometry": {"Coordinates": "139.7671,35.6812"}}]}
        )

        result = geocoding.forward_geocode("東京タワー")

        self.assertEqual(result, (35.6812, 139.7671))

    @patch.dict("os.environ", {"KAKAO_REST_API_KEY": "test-key"})
    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_provider_with_no_result_falls_through(self, mock_urlopen, _sleep):
        mock_urlopen.side_effect = [
            _json_response({"documents": []}),  # Kakao: no match
            _json_response([{"lat": "1.0", "lon": "2.0"}]),  # Nominatim answers
        ]

        self.assertEqual(geocoding.forward_geocode("some place"), (1.0, 2.0))

    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_returns_none_when_nothing_has_an_answer(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response([])
        self.assertIsNone(geocoding.forward_geocode("nowhere in particular"))

    def test_blank_name_returns_none_without_a_request(self):
        self.assertIsNone(geocoding.forward_geocode("   "))

    @patch("photo_grouping.geocoding.time.sleep", return_value=None)
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_malformed_provider_response_falls_through_rather_than_raising(self, mock_urlopen, _sleep):
        mock_urlopen.return_value = _json_response([{"lat": "not-a-number", "lon": "2.0"}])
        self.assertIsNone(geocoding.forward_geocode("weird response"))


class RateLimitTests(GeocodingTestCase):
    @patch("photo_grouping.geocoding.time.monotonic")
    @patch("photo_grouping.geocoding.time.sleep")
    @patch("photo_grouping.geocoding.urllib.request.urlopen")
    def test_enforces_minimum_interval_between_requests(self, mock_urlopen, mock_sleep, mock_monotonic):
        mock_urlopen.return_value = _json_response({"display_name": "x"})
        mock_monotonic.side_effect = [100.0, 100.0, 100.3, 100.3]

        geocoding.reverse_geocode(*PARIS)
        geocoding.reverse_geocode(*PARIS)

        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 0.7, places=3)


if __name__ == "__main__":
    unittest.main()
