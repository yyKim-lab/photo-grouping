"""§4.2 App settings — voice-typing/narrative/UI language, plus the
"Connect your accounts" credential-save routes that replace hand-editing
files under secrets/ for a self-hoster (see web.py's settings section
docstring). All stored in the single-row `app_settings` table (same
pattern as autobio_settings and pending_import_session).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402

DIMS = 512


def _embedding(person: int) -> list[float]:
    vector = [0.0] * DIMS
    vector[person % DIMS] = 1.0
    return vector


class SettingsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db_path = self.tmp / "test.db"
        self.conn = db.connect(self.db_path)
        db.migrate(self.conn)
        web.app.config.update(DB_PATH=self.db_path, TESTING=True)
        self.client = web.app.test_client()
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def _photo(self) -> int:
        from PIL import Image

        self._n += 1
        name = f"p{self._n}.jpg"
        path = self.tmp / name
        Image.new("RGB", (200, 200), color=(70, 70, 70)).save(path, "JPEG")
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at="2026-04-12T10:00:00",
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )

    def _face_cluster(self, n_faces=1, person=0) -> int:
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            for _ in range(n_faces):
                repository.insert_face_instance(
                    self.conn,
                    photo_id=self._photo(),
                    face_cluster_id=cluster_id,
                    bounding_box={"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3},
                    embedding=_embedding(person),
                )
        return cluster_id


class AppSettingsRepositoryTests(SettingsTestCase):
    def test_defaults_to_korean(self):
        self.assertEqual(
            repository.get_app_settings(self.conn),
            {
                "speech_language": "ko-KR",
                "narrative_language": "ko",
                "ui_language": "ko",
                "llm_provider": "",
                "originals_dir": "",
            },
        )

    def test_set_speech_language_persists(self):
        with self.conn:
            repository.set_speech_language(self.conn, "en-US")
        self.assertEqual(repository.get_app_settings(self.conn)["speech_language"], "en-US")

    def test_set_narrative_language_persists(self):
        with self.conn:
            repository.set_narrative_language(self.conn, "fr")
        self.assertEqual(repository.get_app_settings(self.conn)["narrative_language"], "fr")

    def test_set_ui_language_persists(self):
        with self.conn:
            repository.set_ui_language(self.conn, "uk")
        self.assertEqual(repository.get_app_settings(self.conn)["ui_language"], "uk")

    def test_speech_languages_list_has_expected_options(self):
        codes = [code for code, _label in repository.SPEECH_LANGUAGES]
        self.assertEqual(codes, ["ko-KR", "en-US", "ja-JP"])

    def test_languages_list_has_expected_options(self):
        codes = [code for code, _label in repository.LANGUAGES]
        self.assertEqual(codes, ["en", "ko", "ja", "uk", "es", "fr"])


class SettingsPageRoutesTests(SettingsTestCase):
    def test_page_renders_with_default_language_selected(self):
        resp = self.client.get("/settings")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("음성 입력", body)  # ui_language defaults to Korean
        self.assertIn("한국어", body)

    def test_saving_a_language_redirects_and_persists(self):
        resp = self.client.post("/settings/speech-language", data={"speech_language": "ja-JP"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(repository.get_app_settings(self.conn)["speech_language"], "ja-JP")

        resp = self.client.get("/settings")
        body = resp.get_data(as_text=True)
        # The saved language's <option> should now carry `selected`.
        self.assertIn('value="ja-JP" selected', body)

    def test_saving_without_a_language_is_rejected(self):
        resp = self.client.post("/settings/speech-language", data={})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(repository.get_app_settings(self.conn)["speech_language"], "ko-KR")

    def test_saving_narrative_language_redirects_and_persists(self):
        resp = self.client.post("/settings/narrative-language", data={"narrative_language": "es"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(repository.get_app_settings(self.conn)["narrative_language"], "es")

        resp = self.client.get("/settings")
        self.assertIn('value="es" selected', resp.get_data(as_text=True))

    def test_saving_narrative_language_without_a_value_is_rejected(self):
        resp = self.client.post("/settings/narrative-language", data={})
        self.assertEqual(resp.status_code, 400)

    def test_saving_ui_language_redirects_and_persists(self):
        resp = self.client.post("/settings/ui-language", data={"ui_language": "uk"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(repository.get_app_settings(self.conn)["ui_language"], "uk")

        resp = self.client.get("/settings")
        self.assertIn('value="uk" selected', resp.get_data(as_text=True))

    def test_saving_ui_language_without_a_value_is_rejected(self):
        resp = self.client.post("/settings/ui-language", data={})
        self.assertEqual(resp.status_code, 400)


class GoogleCredentialsSaveTests(SettingsTestCase):
    def setUp(self):
        super().setUp()
        self.secret_path = self.tmp / "secrets" / "client_secret.json"
        self._patch = patch("photo_grouping.web.CLIENT_SECRET_PATH", self.secret_path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        super().tearDown()

    def _upload(self, content: str, filename: str = "client_secret.json"):
        from io import BytesIO

        return self.client.post(
            "/settings/google-credentials",
            data={"client_secret_file": (BytesIO(content.encode()), filename)},
            content_type="multipart/form-data",
        )

    def test_valid_json_is_saved_verbatim_and_locked_down(self):
        raw = json.dumps({"installed": {"client_id": "abc", "client_secret": "xyz"}})

        resp = self._upload(raw)

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.secret_path.read_text(), raw)
        # Owner read/write only (0o600) — this is a real credential file.
        self.assertEqual(self.secret_path.stat().st_mode & 0o777, 0o600)

    def test_creates_the_secrets_directory_if_missing(self):
        self.assertFalse(self.secret_path.parent.exists())

        self._upload(json.dumps({"web": {"client_id": "a", "client_secret": "b"}}))

        self.assertTrue(self.secret_path.exists())

    def test_malformed_json_is_rejected_not_written(self):
        resp = self._upload("{not json")

        self.assertEqual(resp.status_code, 400)
        self.assertFalse(self.secret_path.exists())

    def test_json_missing_the_expected_shape_is_rejected(self):
        resp = self._upload(json.dumps({"foo": "bar"}))

        self.assertEqual(resp.status_code, 400)
        self.assertFalse(self.secret_path.exists())

    def test_no_file_chosen_is_rejected(self):
        resp = self.client.post("/settings/google-credentials", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_empty_file_is_rejected(self):
        resp = self._upload("")
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(self.secret_path.exists())


class LLMCredentialsSaveTests(SettingsTestCase):
    def setUp(self):
        super().setUp()
        self.anthropic_path = self.tmp / "secrets" / "anthropic_api_key.txt"
        self.openai_path = self.tmp / "secrets" / "openai_api_key.txt"
        self._patches = [
            patch("photo_grouping.web.ANTHROPIC_KEY_PATH", self.anthropic_path),
            patch("photo_grouping.web.OPENAI_KEY_PATH", self.openai_path),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        super().tearDown()

    def test_anthropic_key_is_saved_and_locked_down(self):
        resp = self.client.post("/settings/anthropic-key", data={"anthropic_api_key": " sk-ant-abc123 "})

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.anthropic_path.read_text(), "sk-ant-abc123\n")
        self.assertEqual(self.anthropic_path.stat().st_mode & 0o777, 0o600)

    def test_openai_key_is_saved(self):
        resp = self.client.post("/settings/openai-key", data={"openai_api_key": "sk-openai-xyz"})

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.openai_path.read_text(), "sk-openai-xyz\n")

    def test_empty_anthropic_key_is_rejected(self):
        resp = self.client.post("/settings/anthropic-key", data={"anthropic_api_key": "  "})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(self.anthropic_path.exists())

    def test_empty_openai_key_is_rejected(self):
        resp = self.client.post("/settings/openai-key", data={"openai_api_key": ""})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(self.openai_path.exists())

    def test_settings_page_reflects_configured_status(self):
        resp = self.client.get("/settings")
        self.assertEqual(resp.status_code, 200)

        self.client.post("/settings/anthropic-key", data={"anthropic_api_key": "sk-ant-abc"})

        resp = self.client.get("/settings")
        # Both files exist independently — Anthropic configured, OpenAI not.
        self.assertTrue(self.anthropic_path.exists())
        self.assertFalse(self.openai_path.exists())


class GeocodingKeySaveTests(SettingsTestCase):
    """Mirrors LLMCredentialsSaveTests above — same save/badge shape,
    for the two optional regional reverse-geocoding keys."""

    def setUp(self):
        super().setUp()
        self.kakao_path = self.tmp / "secrets" / "kakao_rest_api_key.txt"
        self.yahoo_jp_path = self.tmp / "secrets" / "yahoo_jp_client_id.txt"
        self._patches = [
            patch("photo_grouping.web.KAKAO_KEY_PATH", self.kakao_path),
            patch("photo_grouping.web.YAHOO_JP_KEY_PATH", self.yahoo_jp_path),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        super().tearDown()

    def test_kakao_key_is_saved_and_locked_down(self):
        resp = self.client.post("/settings/kakao-key", data={"kakao_rest_api_key": " abc123 "})

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.kakao_path.read_text(), "abc123\n")
        self.assertEqual(self.kakao_path.stat().st_mode & 0o777, 0o600)

    def test_yahoo_jp_client_id_is_saved(self):
        resp = self.client.post("/settings/yahoo-jp-key", data={"yahoo_jp_client_id": "client-xyz"})

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(self.yahoo_jp_path.read_text(), "client-xyz\n")

    def test_empty_kakao_key_is_rejected(self):
        resp = self.client.post("/settings/kakao-key", data={"kakao_rest_api_key": "  "})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(self.kakao_path.exists())

    def test_empty_yahoo_jp_client_id_is_rejected(self):
        resp = self.client.post("/settings/yahoo-jp-key", data={"yahoo_jp_client_id": ""})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(self.yahoo_jp_path.exists())

    def test_settings_pages_reflect_configured_status(self):
        self.client.post("/settings/kakao-key", data={"kakao_rest_api_key": "abc123"})

        for path in ("/settings", "/settings/connect"):
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200, path)
            body = resp.get_data(as_text=True)
            self.assertIn("카카오", body, path)  # ui_language defaults to Korean
        # Kakao configured, Yahoo! JAPAN not.
        self.assertTrue(self.kakao_path.exists())
        self.assertFalse(self.yahoo_jp_path.exists())


class LLMProviderPreferenceTests(SettingsTestCase):
    def test_defaults_to_auto(self):
        self.assertEqual(repository.get_app_settings(self.conn)["llm_provider"], "")

    def test_saving_a_provider_persists(self):
        resp = self.client.post("/settings/llm-provider", data={"llm_provider": "openai"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(repository.get_app_settings(self.conn)["llm_provider"], "openai")

    def test_saving_empty_string_resets_to_auto(self):
        with self.conn:
            repository.set_llm_provider(self.conn, "anthropic")

        self.client.post("/settings/llm-provider", data={"llm_provider": ""})

        self.assertEqual(repository.get_app_settings(self.conn)["llm_provider"], "")

    def test_an_unknown_provider_value_is_rejected(self):
        resp = self.client.post("/settings/llm-provider", data={"llm_provider": "not-a-real-provider"})
        self.assertEqual(resp.status_code, 400)

    @patch("photo_grouping.web.llm.complete")
    def test_generation_uses_the_preferred_provider(self, mock_complete):
        # End-to-end: the Settings-page choice actually reaches llm.complete's
        # `provider` kwarg on a real Autobio generation call.
        from PIL import Image

        path = self.tmp / "p1.jpg"
        Image.new("RGB", (100, 100), color=(50, 50, 50)).save(path, "JPEG")
        with self.conn:
            photo_id = repository.insert_photo(
                self.conn,
                picker_media_id="pmi-1",
                taken_at="2026-04-12T10:00:00",
                original_filename="p1.jpg",
                original_storage_backend="local",
                original_storage_path=str(path),
            )
            repository.set_llm_provider(self.conn, "openai")
        mock_complete.return_value = json.dumps(
            {"segments": [{"text": "x", "source_photo_ids": [photo_id]}]}
        )

        self.client.post("/autobio/generate", data={"date": "2026-04-12"})

        self.assertEqual(mock_complete.call_args.kwargs.get("provider"), "openai")

    @patch("photo_grouping.web.llm.complete")
    def test_generation_leaves_provider_unset_when_preference_is_auto(self, mock_complete):
        from PIL import Image

        path = self.tmp / "p1.jpg"
        Image.new("RGB", (100, 100), color=(50, 50, 50)).save(path, "JPEG")
        with self.conn:
            photo_id = repository.insert_photo(
                self.conn,
                picker_media_id="pmi-1",
                taken_at="2026-04-12T10:00:00",
                original_filename="p1.jpg",
                original_storage_backend="local",
                original_storage_path=str(path),
            )
        mock_complete.return_value = json.dumps(
            {"segments": [{"text": "x", "source_photo_ids": [photo_id]}]}
        )

        self.client.post("/autobio/generate", data={"date": "2026-04-12"})

        self.assertNotIn("provider", mock_complete.call_args.kwargs)


class SpeechLanguagePassedToLabelingPagesTests(SettingsTestCase):
    def test_queue_page_is_reachable_with_no_labeling_work(self):
        # No clusters queued yet — just confirms the route (which now reads
        # app_settings for speech_language) doesn't error out when empty.
        resp = self.client.get("/queue")
        self.assertEqual(resp.status_code, 200)

    def test_queue_page_passes_saved_language_to_the_voice_input_script(self):
        self._face_cluster(2)
        with self.conn:
            repository.set_speech_language(self.conn, "en-US")
        resp = self.client.get("/queue")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("initVoiceInput('name-input', \"en-US\"", resp.get_data(as_text=True))

    def test_photo_detail_page_passes_saved_language_to_the_voice_input_script(self):
        cluster_id = self._face_cluster(1)
        face = repository.representative_face(self.conn, cluster_id)
        with self.conn:
            repository.set_speech_language(self.conn, "ja-JP")
        resp = self.client.get(f"/photo/{face['photo_id']}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("initVoiceInput('add-face-name', \"ja-JP\"", resp.get_data(as_text=True))


class OriginalsDirPreferenceTests(SettingsTestCase):
    """Where imports save their originals (§5's LocalStorageAdapter
    root_dir) — see web.py's _effective_originals_dir() and
    test_import_routes.py for the "an import actually uses it" coverage."""

    def test_defaults_to_empty(self):
        self.assertEqual(repository.get_app_settings(self.conn)["originals_dir"], "")

    def test_saving_a_path_persists(self):
        target = self.tmp / "my-photos"
        resp = self.client.post("/settings/originals-dir", data={"originals_dir": str(target)})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(repository.get_app_settings(self.conn)["originals_dir"], str(target))

    def test_saving_creates_the_directory_if_it_does_not_exist_yet(self):
        target = self.tmp / "not-created-yet" / "nested"
        self.assertFalse(target.exists())

        self.client.post("/settings/originals-dir", data={"originals_dir": str(target)})

        self.assertTrue(target.is_dir())

    def test_saving_empty_string_resets_to_the_default(self):
        with self.conn:
            repository.set_originals_dir(self.conn, str(self.tmp / "somewhere"))

        self.client.post("/settings/originals-dir", data={"originals_dir": ""})

        self.assertEqual(repository.get_app_settings(self.conn)["originals_dir"], "")

    def test_a_relative_path_is_rejected(self):
        resp = self.client.post("/settings/originals-dir", data={"originals_dir": "relative/path"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(repository.get_app_settings(self.conn)["originals_dir"], "")

    def test_settings_page_shows_the_default_when_nothing_is_configured(self):
        resp = self.client.get("/settings")
        self.assertEqual(resp.status_code, 200)
        # "현재 기본 폴더를 사용 중입니다" (part of settings.storage_using_default) —
        # ui_language defaults to Korean.
        self.assertIn("현재 기본 폴더를 사용 중입니다".encode(), resp.data)

    def test_settings_page_shows_the_configured_path_in_the_field(self):
        target = self.tmp / "custom-photos"
        with self.conn:
            repository.set_originals_dir(self.conn, str(target))

        resp = self.client.get("/settings")

        self.assertIn(str(target).encode(), resp.data)


if __name__ == "__main__":
    unittest.main()
