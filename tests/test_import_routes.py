"""Web-triggered import (new — this used to be CLI-only). Google
interaction is mocked at the same boundary the CLI script uses
(google_auth/picker_client), so these test the two-request split and the
error paths, not live OAuth or the Picker UI itself — that needs a human
in a real browser, same as scripts/ingest.py always has.
"""

import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402


class _SyncThread:
    """Stand-in for threading.Thread that runs its target immediately and
    synchronously on .start(), instead of on a real background thread —
    keeps tests deterministic (no sleeping/polling for a real thread to
    finish) while still exercising the exact same code path production
    uses. See ImportContinueTests.setUp()."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


class ImportRouteTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db_path = self.tmp / "test.db"
        self.conn = db.connect(self.db_path)
        db.migrate(self.conn)
        web.app.config.update(DB_PATH=self.db_path, TESTING=True)
        self.client = web.app.test_client()

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()


class PendingImportSessionRepositoryTests(ImportRouteTestCase):
    """Persisted server-side (migration 0009) so an import can be resumed
    from any tab, not just the one that started it — see
    picker_client.delete_session()'s docstring and the README refinements
    section for why the tab-only version broke in practice."""

    def test_nothing_pending_by_default(self):
        self.assertIsNone(repository.get_pending_import_session(self.conn))

    def test_saves_and_reads_back(self):
        with self.conn:
            repository.save_pending_import_session(
                self.conn, session_id="sess-1", picker_uri="https://x/picker"
            )

        pending = repository.get_pending_import_session(self.conn)
        self.assertEqual(pending["session_id"], "sess-1")
        self.assertEqual(pending["picker_uri"], "https://x/picker")

    def test_saving_a_new_session_replaces_the_old_one(self):
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="old", picker_uri="https://x/old")
            repository.save_pending_import_session(self.conn, session_id="new", picker_uri="https://x/new")

        pending = repository.get_pending_import_session(self.conn)
        self.assertEqual(pending["session_id"], "new")

    def test_clear_removes_it(self):
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/picker")
            repository.clear_pending_import_session(self.conn)

        self.assertIsNone(repository.get_pending_import_session(self.conn))

    def test_claim_succeeds_and_removes_the_row(self):
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/picker")
            claimed = repository.claim_pending_import_session(self.conn, "sess-1")

        self.assertTrue(claimed)
        self.assertIsNone(repository.get_pending_import_session(self.conn))

    def test_second_claim_for_the_same_session_fails(self):
        # The two-tabs race this exists for: the picking tab's auto-poll
        # and the /import hub's resume banner both reaching
        # /import/continue for the same session at nearly the same time.
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/picker")
            first = repository.claim_pending_import_session(self.conn, "sess-1")
            second = repository.claim_pending_import_session(self.conn, "sess-1")

        self.assertTrue(first)
        self.assertFalse(second)

    def test_claim_fails_when_nothing_is_pending(self):
        self.assertFalse(repository.claim_pending_import_session(self.conn, "sess-1"))

    def test_claim_fails_for_the_wrong_session_id(self):
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/picker")
            claimed = repository.claim_pending_import_session(self.conn, "sess-2")

        self.assertFalse(claimed)
        # The real pending session is untouched by the mismatched claim.
        self.assertIsNotNone(repository.get_pending_import_session(self.conn))


class StartPageTests(ImportRouteTestCase):
    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_shows_setup_instructions_when_no_credentials(self, mock_path):
        mock_path.exists.return_value = False

        response = self.client.get("/import")

        self.assertEqual(response.status_code, 200)
        # "Google 계정을 연결" (part of import.google_missing_body) — ui_language
        # defaults to Korean; the guidance now links to /settings/connect
        # instead of pointing at README/manual file instructions.
        self.assertIn("Google 계정을 연결".encode(), response.data)
        self.assertIn(b"/settings/connect", response.data)
        self.assertNotIn(b"Start import", response.data)

    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_shows_start_button_when_credentials_exist(self, mock_path):
        mock_path.exists.return_value = True

        response = self.client.get("/import")

        self.assertIn("가져오기 시작".encode(), response.data)  # ui_language defaults to Korean

    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_no_resume_banner_when_nothing_pending(self, mock_path):
        mock_path.exists.return_value = True

        response = self.client.get("/import")

        self.assertNotIn(b"Finish that import", response.data)

    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_shows_resume_banner_for_a_pending_session(self, mock_path):
        mock_path.exists.return_value = True
        with self.conn:
            repository.save_pending_import_session(
                self.conn, session_id="sess-1", picker_uri="https://x/picker"
            )

        response = self.client.get("/import")

        self.assertIn("해당 가져오기 마치기".encode(), response.data)  # ui_language defaults to Korean
        self.assertIn(b"sess-1", response.data)

    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_stale_pending_session_is_dropped_not_offered(self, mock_path):
        # A session left unresumed past _PICKER_UI_STALE_AFTER (observed in
        # practice: Google's picker page itself refuses to open long before
        # the session's own API record expires, see web.py's comment) is
        # already known to be a dead link — quietly cleared instead of
        # offered as "Finish that import".
        mock_path.exists.return_value = True
        with self.conn:
            repository.save_pending_import_session(
                self.conn, session_id="sess-1", picker_uri="https://x/picker"
            )
        self.conn.execute(
            "UPDATE pending_import_session SET created_at = '2000-01-01T00:00:00.000Z'"
        )
        self.conn.commit()

        response = self.client.get("/import")

        self.assertNotIn(b"Finish that import", response.data)
        self.assertIsNone(repository.get_pending_import_session(self.conn))


class ImportStartTests(ImportRouteTestCase):
    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_missing_credentials_is_reported_not_a_crash(self, mock_path):
        mock_path.exists.return_value = False

        response = self.client.post("/import/start")

        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_creates_a_session_and_shows_the_picker_link(self, mock_path, mock_auth, mock_picker):
        mock_path.exists.return_value = True
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.create_session.return_value = {
            "id": "sess-1",
            "pickerUri": "https://photos.google.com/picker/xyz",
        }

        response = self.client.post("/import/start")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"https://photos.google.com/picker/xyz", response.data)
        self.assertIn(b"sess-1", response.data)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_persists_the_session_server_side(self, mock_path, mock_auth, mock_picker):
        mock_path.exists.return_value = True
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.create_session.return_value = {
            "id": "sess-1",
            "pickerUri": "https://photos.google.com/picker/xyz",
        }

        self.client.post("/import/start")

        pending = repository.get_pending_import_session(self.conn)
        self.assertEqual(pending["session_id"], "sess-1")

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    @patch("photo_grouping.web.CLIENT_SECRET_PATH")
    def test_get_also_works_not_just_post(self, mock_path, mock_auth, mock_picker):
        # The template links here via a plain <a target="_blank"> GET link,
        # not a form POST — a method="post" form with target="_blank" was
        # observed degrading to a bodyless GET and 405ing instead of
        # opening a new tab at all, so GET is what's actually used now.
        mock_path.exists.return_value = True
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.create_session.return_value = {
            "id": "sess-1",
            "pickerUri": "https://photos.google.com/picker/xyz",
        }

        response = self.client.get("/import/start")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"sess-1", response.data)


class ImportContinueTests(ImportRouteTestCase):
    """Ingestion itself now runs on a background thread (started from
    import_continue) so the browser gets a live progress page instead of
    blocking on a slow POST — see import_progress.html. Patching
    threading.Thread to run synchronously keeps these tests deterministic
    while still exercising the exact same code the real thread runs."""

    def setUp(self):
        super().setUp()
        self._thread_patch = patch("photo_grouping.web.threading.Thread", _SyncThread)
        self._thread_patch.start()

    def tearDown(self):
        self._thread_patch.stop()
        super().tearDown()

    def test_missing_session_id_is_rejected(self):
        response = self.client.post("/import/continue", data={})
        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_not_yet_picked_shows_a_retry_prompt(self, mock_auth, mock_picker):
        mock_auth.get_access_token.return_value = "fake-token"
        # Note: no "pickerUri" key here on purpose — per Google's docs only
        # mediaItemsSet/pollingConfig are guaranteed on GET /v1/sessions/{id},
        # so this response must not need to be read for pickerUri at all.
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": False}

        response = self.client.post(
            "/import/continue",
            data={"session_id": "sess-1", "picker_uri": "https://photos.google.com/picker/xyz"},
        )

        self.assertEqual(response.status_code, 200)
        # "아직 선택을 마치지 않았습니다" ("haven't finished picking yet") — ui_language
        # defaults to Korean.
        self.assertIn("아직 선택을 마치지 않았습니다".encode(), response.data)
        self.assertIn(b"https://photos.google.com/picker/xyz", response.data)
        mock_picker.list_media_items.assert_not_called()

    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_ready_shows_a_progress_page_immediately(self, mock_auth, mock_picker, mock_face):
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = [
            {"id": "m1", "mediaFile": {"filename": "a.jpg", "baseUrl": "https://x/a"}},
        ]
        mock_picker.fetch_original_bytes.return_value = self._fake_jpeg_bytes()
        mock_face.detect_faces_in_bytes.return_value = []

        response = self.client.post("/import/continue", data={"session_id": "sess-1"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("사진을 가져오는 중".encode(), response.data)  # ui_language defaults to Korean
        self.assertIn(b"sess-1", response.data)

    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_ingestion_completes_and_result_is_fetchable(self, mock_auth, mock_picker, mock_face):
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = [
            {"id": "m1", "mediaFile": {"filename": "a.jpg", "baseUrl": "https://x/a"}},
        ]
        mock_picker.fetch_original_bytes.return_value = self._fake_jpeg_bytes()
        mock_face.detect_faces_in_bytes.return_value = []

        self.client.post("/import/continue", data={"session_id": "sess-1"})
        # The _SyncThread double runs the ingestion inline during the POST
        # above, so the job is already "done" by the time we check.
        progress = self.client.get("/import/progress", query_string={"job_id": "sess-1"})
        result = self.client.get("/import/result", query_string={"job_id": "sess-1"})

        self.assertEqual(progress.get_json()["state"], "done")
        self.assertIn("가져오기 완료".encode(), result.data)  # ui_language defaults to Korean
        mock_picker.delete_session.assert_called_once()
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 1)

    @patch("photo_grouping.web.ingestion.suggest_location_names")
    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_location_suggestions_are_backfilled_after_ingestion(
        self, mock_auth, mock_picker, mock_face, mock_suggest
    ):
        # Real bug hit in practice: this only used to happen via the
        # terminal-only scripts/ingest.py, never from the web import
        # routes — every web-imported photo's place clusters sat with no
        # geocoded/OCR suggestion forever. Confirms the fix actually
        # wires it in, without needing real GPS-bearing fixtures or a
        # real geocoding call (mocked out entirely here).
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = [
            {"id": "m1", "mediaFile": {"filename": "a.jpg", "baseUrl": "https://x/a"}},
        ]
        mock_picker.fetch_original_bytes.return_value = self._fake_jpeg_bytes()
        mock_face.detect_faces_in_bytes.return_value = []

        self.client.post("/import/continue", data={"session_id": "sess-1"})

        mock_suggest.assert_called_once()

    @patch("photo_grouping.web.storage.get_adapter")
    @patch("photo_grouping.web.ingestion.suggest_location_names")
    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_a_configured_originals_dir_is_used_instead_of_the_default(
        self, mock_auth, mock_picker, mock_face, mock_suggest, mock_get_adapter
    ):
        # web.py used to hardcode DEFAULT_ORIGINALS_DIR for every import
        # regardless of any setting — confirms _effective_originals_dir()
        # actually reaches storage.get_adapter()'s root_dir argument.
        custom_dir = self.tmp / "my-photos"
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
            repository.set_originals_dir(self.conn, str(custom_dir))
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = [
            {"id": "m1", "mediaFile": {"filename": "a.jpg", "baseUrl": "https://x/a"}},
        ]
        mock_picker.fetch_original_bytes.return_value = self._fake_jpeg_bytes()
        mock_face.detect_faces_in_bytes.return_value = []
        mock_get_adapter.return_value.save_original.return_value = ("local", str(custom_dir / "a.jpg"))

        self.client.post("/import/continue", data={"session_id": "sess-1"})

        mock_get_adapter.assert_called_once_with("local", custom_dir)

    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_success_clears_the_pending_session(self, mock_auth, mock_picker, mock_face):
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = []
        mock_face.detect_faces_in_bytes.return_value = []

        self.client.post("/import/continue", data={"session_id": "sess-1"})

        self.assertIsNone(repository.get_pending_import_session(self.conn))

    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_reports_the_gps_backfill_is_a_separate_step(self, mock_auth, mock_picker, mock_face):
        # Web import only brings in photos + faces; GPS needs a Takeout
        # export the browser can't provide (see README's constraint note).
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = []
        mock_face.detect_faces_in_bytes.return_value = []

        self.client.post("/import/continue", data={"session_id": "sess-1"})
        result = self.client.get("/import/result", query_string={"job_id": "sess-1"})
        body = result.get_data(as_text=True)

        # The short note is always there; the in-app "how do I do this?"
        # disclosure carries the real step-by-step guidance (a Takeout
        # link, and the actual runnable commands with this install's real
        # path filled in) so people aren't sent to the README for it.
        self.assertIn("backfill-gps", body)
        self.assertIn('<details class="howto">', body)
        self.assertIn('href="https://takeout.google.com"', body)
        self.assertIn("cluster-locations", body)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_expired_session_is_reported_not_a_crash(self, mock_auth, mock_picker):
        import urllib.error
        from unittest.mock import MagicMock

        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.side_effect = urllib.error.HTTPError(
            url="https://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=None
        )

        response = self.client.post("/import/continue", data={"session_id": "sess-1"})

        self.assertEqual(response.status_code, 410)
        self.assertIsNone(repository.get_pending_import_session(self.conn))

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_stale_pending_session_is_reported_as_expired_not_re_offered(self, mock_auth, mock_picker):
        # Real bug hit in practice: Google's GET /v1/sessions/{id} kept
        # returning 200/mediaItemsSet=false for a session left unresumed
        # ~21h, but its pickerUri had already stopped opening (Google's own
        # "Couldn't open Google Photos" page). Re-serving that same dead
        # link forever is worse than just saying it expired.
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        self.conn.execute(
            "UPDATE pending_import_session SET created_at = '2000-01-01T00:00:00.000Z'"
        )
        self.conn.commit()
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": False}

        response = self.client.post(
            "/import/continue",
            data={"session_id": "sess-1", "picker_uri": "https://x/p"},
        )

        self.assertEqual(response.status_code, 410)
        self.assertIn(b"too long", response.data)
        self.assertIsNone(repository.get_pending_import_session(self.conn))

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_non_404_errors_still_propagate(self, mock_auth, mock_picker):
        import urllib.error
        from unittest.mock import MagicMock

        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.side_effect = urllib.error.HTTPError(
            url="https://x", code=500, msg="Server Error", hdrs=MagicMock(), fp=None
        )

        with self.assertRaises(urllib.error.HTTPError):
            self.client.post("/import/continue", data={"session_id": "sess-1"})

    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_second_concurrent_request_for_the_same_session_is_handled_gracefully(
        self, mock_auth, mock_picker, mock_face
    ):
        # The exact race observed in practice: the picking tab's auto-poll
        # and the /import hub's "Finish that import" banner both reach
        # this route for the same session — nothing pending means someone
        # (very likely: this same session, from the other tab) already
        # claimed and processed it.
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = []
        mock_face.detect_faces_in_bytes.return_value = []
        # No save_pending_import_session() call — simulates the row
        # already having been claimed (and cleared) by the "winner".

        response = self.client.post("/import/continue", data={"session_id": "sess-1"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("이미 처리됨".encode(), response.data)  # ui_language defaults to Korean
        mock_picker.list_media_items.assert_not_called()

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_list_media_items_404_is_handled_gracefully_not_a_crash(self, mock_auth, mock_picker):
        import urllib.error
        from unittest.mock import MagicMock

        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.side_effect = urllib.error.HTTPError(
            url="https://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=None
        )

        response = self.client.post("/import/continue", data={"session_id": "sess-1"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("이미 처리됨".encode(), response.data)  # ui_language defaults to Korean

    def _fake_jpeg_bytes(self) -> bytes:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (100, 100), color=(50, 50, 50)).save(buf, "JPEG")
        return buf.getvalue()


class ImportProgressAndResultTests(ImportRouteTestCase):
    """/import/progress and /import/result back the progress page's
    polling loop directly, independent of the full /import/continue flow
    that normally creates the underlying job."""

    def setUp(self):
        super().setUp()
        self._thread_patch = patch("photo_grouping.web.threading.Thread", _SyncThread)
        self._thread_patch.start()

    def tearDown(self):
        self._thread_patch.stop()
        super().tearDown()

    def test_progress_missing_session_id_is_rejected(self):
        response = self.client.get("/import/progress")
        self.assertEqual(response.status_code, 400)

    def test_progress_unknown_session_is_404(self):
        response = self.client.get("/import/progress", query_string={"job_id": "nope"})
        self.assertEqual(response.status_code, 404)

    def test_result_missing_session_id_is_rejected(self):
        response = self.client.get("/import/result")
        self.assertEqual(response.status_code, 400)

    def test_result_unknown_session_is_404(self):
        response = self.client.get("/import/result", query_string={"job_id": "nope"})
        self.assertEqual(response.status_code, 404)

    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_result_while_still_running_is_404_not_a_blank_page(self, mock_auth, mock_picker, mock_face):
        # threading.Thread is patched to run synchronously in this class,
        # so "still running" can't naturally happen here — this instead
        # verifies the guard directly via a job in that state.
        job = web._ImportJob(total=1, source="google")
        with web._import_jobs_lock:
            web._import_jobs["sess-running"] = job

        response = self.client.get("/import/result", query_string={"job_id": "sess-running"})

        self.assertEqual(response.status_code, 404)
        # Not destroyed by the premature call — still retrievable once it
        # actually finishes.
        with web._import_jobs_lock:
            self.assertIn("sess-running", web._import_jobs)
            del web._import_jobs["sess-running"]

    @patch("photo_grouping.web.face_embeddings")
    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_a_failure_mid_ingestion_is_reported_not_a_crash(self, mock_auth, mock_picker, mock_face):
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}
        mock_picker.list_media_items.return_value = [
            {"id": "m1", "mediaFile": {"filename": "a.jpg", "baseUrl": "https://x/a"}},
        ]
        mock_picker.fetch_original_bytes.side_effect = RuntimeError("simulated disk failure")

        self.client.post("/import/continue", data={"session_id": "sess-1"})
        progress = self.client.get("/import/progress", query_string={"job_id": "sess-1"})

        # A per-photo failure inside ingest_picked_items() is already
        # caught there (goes into result.failed) rather than raising — an
        # error state here means something escaped that, e.g. a problem in
        # delete_session or the surrounding thread setup itself.
        self.assertEqual(progress.get_json()["state"], "done")

    def test_result_is_removed_after_being_fetched_once(self):
        from photo_grouping import ingestion

        job = web._ImportJob(total=0, source="google")
        job.state = "done"
        job.result = ingestion.IngestionResult()
        with web._import_jobs_lock:
            web._import_jobs["sess-once"] = job

        first = self.client.get("/import/result", query_string={"job_id": "sess-once"})
        second = self.client.get("/import/result", query_string={"job_id": "sess-once"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 404)


class ImportStatusTests(ImportRouteTestCase):
    """Backs the picking page's auto-poll JS — see import_picking.html."""

    def test_missing_session_id_is_rejected(self):
        response = self.client.get("/import/status")
        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_expired_session_reports_expired_not_a_crash(self, mock_auth, mock_picker):
        import urllib.error
        from unittest.mock import MagicMock

        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.side_effect = urllib.error.HTTPError(
            url="https://x", code=404, msg="Not Found", hdrs=MagicMock(), fp=None
        )

        response = self.client.get("/import/status", query_string={"session_id": "sess-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ready": False, "expired": True})

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_reports_ready_true_once_media_items_set(self, mock_auth, mock_picker):
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": True}

        response = self.client.get("/import/status", query_string={"session_id": "sess-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ready": True})

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_reports_ready_false_while_still_picking(self, mock_auth, mock_picker):
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": False}

        response = self.client.get("/import/status", query_string={"session_id": "sess-1"})

        self.assertEqual(response.get_json(), {"ready": False})

    @patch("photo_grouping.web.picker_client")
    @patch("photo_grouping.web.google_auth")
    def test_stale_pending_session_reports_expired_not_endless_polling(self, mock_auth, mock_picker):
        # A picking tab left open past _PICKER_UI_STALE_AFTER would
        # otherwise poll "not ready" forever against a picker link Google
        # has already abandoned — this stops the auto-poll loop the same
        # way a real 404 does.
        with self.conn:
            repository.save_pending_import_session(self.conn, session_id="sess-1", picker_uri="https://x/p")
        self.conn.execute(
            "UPDATE pending_import_session SET created_at = '2000-01-01T00:00:00.000Z'"
        )
        self.conn.commit()
        mock_auth.get_access_token.return_value = "fake-token"
        mock_picker.get_session.return_value = {"id": "sess-1", "mediaItemsSet": False}

        response = self.client.get("/import/status", query_string={"session_id": "sess-1"})

        self.assertEqual(response.get_json(), {"ready": False, "expired": True})
        self.assertIsNone(repository.get_pending_import_session(self.conn))


class ImportLocalTests(ImportRouteTestCase):
    """Local-device import — no Google/OAuth involved, so no mocking of
    google_auth/picker_client needed here at all. Runs through the same
    background-job/progress-page machinery as the Google flow (see
    ImportContinueTests), just keyed by a generated job id instead of a
    Google session_id — hence the same threading.Thread patch."""

    def setUp(self):
        super().setUp()
        self._thread_patch = patch("photo_grouping.web.threading.Thread", _SyncThread)
        self._thread_patch.start()

    def tearDown(self):
        self._thread_patch.stop()
        super().tearDown()

    def test_form_page_renders(self):
        response = self.client.get("/import/local")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Import from this device", response.data)

    def test_no_files_selected_is_reported_not_a_crash(self):
        response = self.client.post("/import/local", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.face_embeddings")
    def test_upload_shows_a_progress_page_immediately(self, mock_face):
        mock_face.detect_faces_in_bytes.return_value = []

        response = self.client.post(
            "/import/local",
            data={"files": [(BytesIO(self._fake_jpeg_bytes()), "a.jpg")]},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("사진을 가져오는 중".encode(), response.data)  # ui_language defaults to Korean

    @patch("photo_grouping.web.face_embeddings")
    def test_uploads_are_ingested_and_summarized(self, mock_face):
        mock_face.detect_faces_in_bytes.return_value = []

        response = self.client.post(
            "/import/local",
            data={"files": [(BytesIO(self._fake_jpeg_bytes()), "a.jpg")]},
            content_type="multipart/form-data",
        )
        result = self.client.get(
            "/import/result", query_string={"job_id": self._extract_job_id(response)}
        )

        self.assertIn("가져오기 완료".encode(), result.data)  # ui_language defaults to Korean
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 1)

    @patch("photo_grouping.web.ingestion.suggest_location_names")
    @patch("photo_grouping.web.face_embeddings")
    def test_location_suggestions_are_backfilled_after_ingestion(self, mock_face, mock_suggest):
        # Same fix as the Google-import path — see the matching test in
        # ImportContinueTests for the full story.
        mock_face.detect_faces_in_bytes.return_value = []

        self.client.post(
            "/import/local",
            data={"files": [(BytesIO(self._fake_jpeg_bytes()), "a.jpg")]},
            content_type="multipart/form-data",
        )

        mock_suggest.assert_called_once()

    @patch("photo_grouping.web.storage.get_adapter")
    @patch("photo_grouping.web.face_embeddings")
    def test_a_configured_originals_dir_is_used_instead_of_the_default(self, mock_face, mock_get_adapter):
        # Same fix as the Google-import path — see the matching test in
        # ImportContinueTests for the full story.
        custom_dir = self.tmp / "my-photos"
        with self.conn:
            repository.set_originals_dir(self.conn, str(custom_dir))
        mock_face.detect_faces_in_bytes.return_value = []
        mock_get_adapter.return_value.save_original.return_value = ("local", str(custom_dir / "a.jpg"))

        self.client.post(
            "/import/local",
            data={"files": [(BytesIO(self._fake_jpeg_bytes()), "a.jpg")]},
            content_type="multipart/form-data",
        )

        mock_get_adapter.assert_called_once_with("local", custom_dir)

    @patch("photo_grouping.web.face_embeddings")
    def test_gps_note_differs_from_the_google_flow(self, mock_face):
        mock_face.detect_faces_in_bytes.return_value = []

        response = self.client.post(
            "/import/local",
            data={"files": [(BytesIO(self._fake_jpeg_bytes()), "a.jpg")]},
            content_type="multipart/form-data",
        )
        result = self.client.get(
            "/import/result", query_string={"job_id": self._extract_job_id(response)}
        )
        body = result.get_data(as_text=True)

        self.assertNotIn("backfill-gps", body)
        self.assertNotIn('<details class="howto">', body)

    @patch("photo_grouping.web.face_embeddings")
    def test_multiple_files_all_import(self, mock_face):
        mock_face.detect_faces_in_bytes.return_value = []

        response = self.client.post(
            "/import/local",
            data={
                "files": [
                    (BytesIO(self._fake_jpeg_bytes(color=(50, 50, 50))), "a.jpg"),
                    (BytesIO(self._fake_jpeg_bytes(color=(80, 20, 200))), "b.jpg"),
                ]
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 2)

    @patch("photo_grouping.web.face_embeddings")
    def test_progress_is_reported_via_on_progress_callback(self, mock_face):
        mock_face.detect_faces_in_bytes.return_value = []

        response = self.client.post(
            "/import/local",
            data={"files": [(BytesIO(self._fake_jpeg_bytes()), "a.jpg")]},
            content_type="multipart/form-data",
        )
        job_id = self._extract_job_id(response)
        progress = self.client.get("/import/progress", query_string={"job_id": job_id})

        data = progress.get_json()
        self.assertEqual(data["state"], "done")
        self.assertEqual(data["current"], 1)
        self.assertEqual(data["total"], 1)
        self.assertIn("a.jpg", data["last"])

    def _extract_job_id(self, response) -> str:
        import re

        match = re.search(rb'var jobId = "([^"]+)"', response.data)
        self.assertIsNotNone(match, "no job id found in the progress page response")
        return match.group(1).decode()

    def _fake_jpeg_bytes(self, color=(50, 50, 50)) -> bytes:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (100, 100), color=color).save(buf, "JPEG")
        return buf.getvalue()


if __name__ == "__main__":
    unittest.main()
