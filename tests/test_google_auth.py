"""Tests for the pieces of google_auth.py that don't require an actual
browser/network round trip: client-secret parsing and PKCE generation.
The interactive loopback flow itself is exercised by running
scripts/picker_spike.py by hand against real Google credentials.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import google_auth  # noqa: E402


class LoadClientCredentialsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, payload: dict) -> Path:
        path = Path(self._tmp.name) / "client_secret.json"
        path.write_text(json.dumps(payload))
        return path

    def test_parses_installed_app_format(self):
        path = self._write({"installed": {"client_id": "id-1", "client_secret": "secret-1"}})
        creds = google_auth.load_client_credentials(path)
        self.assertEqual(creds.client_id, "id-1")
        self.assertEqual(creds.client_secret, "secret-1")

    def test_parses_web_format(self):
        path = self._write({"web": {"client_id": "id-2", "client_secret": "secret-2"}})
        creds = google_auth.load_client_credentials(path)
        self.assertEqual(creds.client_id, "id-2")

    def test_rejects_unrecognized_format(self):
        path = self._write({"something_else": {}})
        with self.assertRaises(ValueError):
            google_auth.load_client_credentials(path)


class PkcePairTests(unittest.TestCase):
    def test_verifier_and_challenge_are_distinct_and_urlsafe(self):
        verifier, challenge = google_auth._make_pkce_pair()
        self.assertNotEqual(verifier, challenge)
        for value in (verifier, challenge):
            self.assertTrue(value)
            self.assertNotIn("+", value)
            self.assertNotIn("/", value)
            self.assertNotIn("=", value)  # padding stripped, per spec's S256 method

    def test_pairs_are_not_reused_across_calls(self):
        pair1 = google_auth._make_pkce_pair()
        pair2 = google_auth._make_pkce_pair()
        self.assertNotEqual(pair1, pair2)


def _fake_id_token(payload: dict) -> str:
    import base64 as b64

    def seg(d: dict) -> str:
        return b64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    # Real signature not needed — _email_from_id_token never verifies one
    # (see its docstring for why: this token always arrives straight from
    # Google's own token endpoint, nothing untrusted to verify against).
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.fake-signature"


class EmailFromIdTokenTests(unittest.TestCase):
    def test_extracts_the_email_claim(self):
        token = _fake_id_token({"email": "someone@example.com", "sub": "123"})
        self.assertEqual(google_auth._email_from_id_token(token), "someone@example.com")

    def test_missing_email_claim_returns_none(self):
        token = _fake_id_token({"sub": "123"})
        self.assertIsNone(google_auth._email_from_id_token(token))

    def test_malformed_token_returns_none_not_a_crash(self):
        self.assertIsNone(google_auth._email_from_id_token("not.a.jwt!!!"))
        self.assertIsNone(google_auth._email_from_id_token(""))


class GetCachedEmailTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.token_path = self.tmp / "token.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_cache_file_returns_none(self):
        self.assertIsNone(google_auth.get_cached_email(self.token_path))

    def test_reads_the_cached_email(self):
        self.token_path.write_text(json.dumps({"access_token": "x", "email": "me@example.com"}))
        self.assertEqual(google_auth.get_cached_email(self.token_path), "me@example.com")

    def test_cache_from_before_the_email_scope_existed_returns_none_not_a_crash(self):
        # An older token.json — signed in before `email` was added to
        # SCOPE — has no "email" key at all. Should degrade gracefully,
        # not force a re-auth just to keep the app working.
        self.token_path.write_text(json.dumps({"access_token": "x", "refresh_token": "y"}))
        self.assertIsNone(google_auth.get_cached_email(self.token_path))

    def test_corrupted_cache_file_returns_none_not_a_crash(self):
        self.token_path.write_text("{not valid json")
        self.assertIsNone(google_auth.get_cached_email(self.token_path))


if __name__ == "__main__":
    unittest.main()
