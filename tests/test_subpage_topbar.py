"""The contextual top app bar (back arrow + generic page-type title) on
sub-pages — see _subpage_context()/_TAB_PRIMARY_ENDPOINT in web.py.

Top-level pages (the one landing page per nav tab) keep the brand + full
nav row unchanged; every other page mapped in _NAV_SECTIONS is, by
definition, a sub-page and gets back+title instead.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402
from _attr_helpers import get_attr_by_class  # noqa: E402


class TopbarTestCase(unittest.TestCase):
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
        Image.new("RGB", (200, 200), color=(80, 80, 80)).save(path, "JPEG")
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at=f"2026-04-{10 + self._n:02d}T10:00:00",
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )

    def _face_cluster(self) -> int:
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn,
                photo_id=self._photo(),
                face_cluster_id=cluster_id,
                bounding_box={"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3},
                embedding=[0.1] * 512,
            )
        return cluster_id


class TopLevelPagesKeepTheBrandTests(TopbarTestCase):
    """The landing page for each nav tab is unaffected — no back button,
    brand text still shows."""

    def _assert_top_level(self, path):
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('class="brand"', body, path)
        self.assertNotIn('class="back-btn"', body, path)
        self.assertNotIn('header class="subpage"', body, path)

    def test_index(self):
        self._assert_top_level("/")

    def test_timeline(self):
        self._assert_top_level("/timeline")

    def test_events_index(self):
        self._assert_top_level("/events")

    def test_queue(self):
        self._assert_top_level("/queue")

    def test_autobio_daily_index(self):
        self._assert_top_level("/autobio/daily")

    def test_autobio_index(self):
        self._assert_top_level("/autobio")

    def test_import_start_page(self):
        self._assert_top_level("/import")

    def test_settings_page(self):
        self._assert_top_level("/settings")


class SubpagesGetTheContextualTopbarTests(TopbarTestCase):
    def _assert_subpage(self, path, expected_title, back_fallback):
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('header class="subpage"', body, path)
        self.assertIn('class="back-btn"', body, path)
        self.assertNotIn('class="brand"', body, path)
        self.assertIn(f">{expected_title}<", body, path)
        self.assertIn(back_fallback, body, path)
        # Regression check: the back button's onclick used to get silently
        # truncated by an unescaped quote from `tojson` output landing
        # inside this double-quoted attribute (fixed via `| forceescape`).
        # Parse it the way a browser would, not via substring matching.
        onclick = get_attr_by_class(body, "back-btn", "onclick")
        self.assertIn("history.back()", onclick, path)
        self.assertTrue(onclick.rstrip().endswith("}"), (path, onclick))
        self.assertIn(f'location.href = "{back_fallback}"', onclick, path)

    def test_cluster_detail_face(self):
        cluster_id = self._face_cluster()
        self._assert_subpage(f"/cluster/face/{cluster_id}", "인물", "/")

    def test_cluster_detail_place(self):
        photo_id = self._photo()
        with self.conn:
            cluster_id = repository.set_photo_location_by_name_without_coordinates(
                self.conn, photo_id, "우리집"
            )
        self._assert_subpage(f"/cluster/place/{cluster_id}", "장소", "/")

    def test_event_detail(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
        self._assert_subpage(f"/event/{event_id}", "그룹", "/events")

    def test_photo_detail(self):
        photo_id = self._photo()
        self._assert_subpage(f"/photo/{photo_id}", "사진", "/timeline")

    def test_autobio_entry(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-04-11", segments=[], draft_text="Draft.", has_unlabeled=False
            )
        self._assert_subpage("/autobio/2026-04-11", "일기", "/autobio/daily")

    def test_autobio_summary_view(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-04-01", end_date="2026-04-11",
                source_entry_ids=[], text="A narrative.",
            )
        self._assert_subpage("/autobio/summary/2026-04-01/2026-04-11", "자서전", "/autobio")

    def test_excluded(self):
        self._assert_subpage("/excluded", "숨김", "/")

    def test_import_local_form(self):
        self._assert_subpage("/import/local", "가져오기", "/import")

    def test_settings_connect_page(self):
        self._assert_subpage("/settings/connect", "설정", "/settings")


if __name__ == "__main__":
    unittest.main()
