"""UI language — the app's own menus/buttons/labels (see i18n.py),
independent of narrative_language (what language Diary/Autobio text is
drafted in, covered in test_autobio.py/test_settings.py instead)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, i18n, repository, web  # noqa: E402


class TranslationLookupTests(unittest.TestCase):
    def test_looks_up_the_requested_language(self):
        self.assertEqual(i18n.t("nav.photos", "ko"), "사진")
        self.assertEqual(i18n.t("nav.photos", "fr"), "Photos")

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(i18n.t("nav.photos", "xx"), i18n.t("nav.photos", "en"))

    def test_unknown_key_falls_back_to_the_key_itself(self):
        self.assertEqual(i18n.t("nonexistent.key", "ko"), "nonexistent.key")

    def test_format_placeholder_is_substituted(self):
        self.assertIn("포토앨범", i18n.t("shared.merged_notice", "ko", name="포토앨범"))

    def test_all_languages_define_the_same_keys_as_english(self):
        english_keys = set(i18n.TRANSLATIONS["en"])
        for lang, table in i18n.TRANSLATIONS.items():
            with self.subTest(lang=lang):
                self.assertEqual(set(table), english_keys)


class NavTranslationRouteTests(unittest.TestCase):
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

    def test_nav_defaults_to_korean(self):
        resp = self.client.get("/")
        body = resp.get_data(as_text=True)
        self.assertIn(">사진<", body)
        self.assertIn(">그룹<", body)

    def test_nav_switches_with_the_ui_language_setting(self):
        with self.conn:
            repository.set_ui_language(self.conn, "fr")
        resp = self.client.get("/")
        body = resp.get_data(as_text=True)
        self.assertIn(">Photos<", body)
        self.assertIn(">Groupes<", body)

    def test_unsaved_lightbox_labels_translate_too(self):
        with self.conn:
            repository.set_ui_language(self.conn, "es")
        resp = self.client.get("/")
        self.assertIn('aria-label="Cerrar"', resp.get_data(as_text=True))


class NavActiveStateTests(unittest.TestCase):
    """The 'active' class marking the current nav tab (see web.py's
    _NAV_SECTIONS / inject_i18n's nav_active) — the CSS for it existed
    from the start, but nothing ever set the class until now."""

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

    def test_albums_is_active_on_the_index_page(self):
        body = self.client.get("/").get_data(as_text=True)
        self.assertIn('class="active">앨범<', body)
        self.assertNotIn('class="active">사진<', body)

    def test_photos_is_active_on_the_timeline_page(self):
        body = self.client.get("/timeline").get_data(as_text=True)
        self.assertIn('class="active">사진<', body)
        self.assertNotIn('class="active">앨범<', body)

    def test_groups_is_active_on_the_events_page(self):
        body = self.client.get("/events").get_data(as_text=True)
        self.assertIn('class="active">그룹<', body)

    def test_settings_is_active_on_the_connect_accounts_page(self):
        # A sub-page under Settings should still highlight Settings.
        body = self.client.get("/settings/connect").get_data(as_text=True)
        self.assertIn('class="active">설정<', body)

    def test_no_tab_is_active_on_an_unmapped_page(self):
        body = self.client.get("/review-duplicates").get_data(as_text=True)
        self.assertNotIn('class="active"', body)


if __name__ == "__main__":
    unittest.main()
