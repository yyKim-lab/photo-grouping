"""Events — new, not in the original spec: a purely user-created grouping
of photos, unlike FaceCluster/LocationCluster which a detector proposes.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402


class EventTestCase(unittest.TestCase):
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
        Image.new("RGB", (200, 200), color=(90, 90, 90)).save(path, "JPEG")
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at=f"2026-04-{10 + self._n:02d}T10:00:00",
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )


class CreateAndAddTests(EventTestCase):
    def test_create_event_requires_a_name(self):
        with self.assertRaises(ValueError):
            repository.create_event(self.conn, "   ")

    def test_get_or_create_reuses_an_existing_event_by_name(self):
        with self.conn:
            first = repository.get_or_create_event(self.conn, "가평 여행")
            second = repository.get_or_create_event(self.conn, "가평 여행")

        self.assertEqual(first, second)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM event").fetchone()["c"], 1)

    def test_add_photos_is_idempotent(self):
        photo_id = self._photo()
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            added_first = repository.add_photos_to_event(self.conn, event_id, [photo_id])
            added_second = repository.add_photos_to_event(self.conn, event_id, [photo_id])

        self.assertEqual(added_first, 1)
        self.assertEqual(added_second, 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) c FROM photo_event WHERE event_id = ?", (event_id,)).fetchone()["c"],
            1,
        )

    def test_a_photo_can_belong_to_more_than_one_event(self):
        photo_id = self._photo()
        with self.conn:
            trip = repository.create_event(self.conn, "가평 여행")
            birthday = repository.create_event(self.conn, "생일")
            repository.add_photos_to_event(self.conn, trip, [photo_id])
            repository.add_photos_to_event(self.conn, birthday, [photo_id])

        events = repository.events_for_photo(self.conn, photo_id)
        self.assertEqual({e["name"] for e in events}, {"가평 여행", "생일"})

    def test_remove_photo_from_event(self):
        photo_id = self._photo()
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.add_photos_to_event(self.conn, event_id, [photo_id])
            repository.remove_photo_from_event(self.conn, event_id, photo_id)

        self.assertEqual(repository.event_detail(self.conn, event_id)["photos"], [])


class EventDetailTests(EventTestCase):
    def test_returns_none_for_unknown_event(self):
        self.assertIsNone(repository.event_detail(self.conn, 9999))

    def test_lists_photos_ordered_by_taken_at(self):
        p1, p2 = self._photo(), self._photo()  # p2 has a later taken_at
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.add_photos_to_event(self.conn, event_id, [p2, p1])

        photos = repository.event_detail(self.conn, event_id)["photos"]
        self.assertEqual([p["photo_id"] for p in photos], [p1, p2])

    def test_rename(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "old")
            repository.rename_event(self.conn, event_id, "new")

        self.assertEqual(repository.event_detail(self.conn, event_id)["event"]["name"], "new")

    def test_delete_removes_the_event_and_its_photo_links(self):
        photo_id = self._photo()
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.add_photos_to_event(self.conn, event_id, [photo_id])
            repository.delete_event(self.conn, event_id)

        self.assertIsNone(repository.event_detail(self.conn, event_id))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo_event").fetchone()["c"], 0)
        # The photo itself is untouched.
        self.assertIsNotNone(repository.photo_detail(self.conn, photo_id))

    def test_set_description(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.set_event_description(self.conn, event_id, "재밌었다")

        self.assertEqual(repository.event_detail(self.conn, event_id)["event"]["description"], "재밌었다")


class ListEventsTests(EventTestCase):
    def test_ordered_by_photo_count_descending(self):
        p1, p2, p3 = self._photo(), self._photo(), self._photo()
        with self.conn:
            small = repository.create_event(self.conn, "작은")
            big = repository.create_event(self.conn, "큰")
            repository.add_photos_to_event(self.conn, small, [p1])
            repository.add_photos_to_event(self.conn, big, [p2, p3])

        events = repository.list_events(self.conn)
        self.assertEqual([e["id"] for e in events], [big, small])


class RouteTests(EventTestCase):
    def test_events_index_renders(self):
        response = self.client.get("/events")
        self.assertEqual(response.status_code, 200)

    def test_bulk_add_creates_and_assigns(self):
        p1, p2 = self._photo(), self._photo()

        response = self.client.post(
            "/events/bulk-add",
            data={"event_name": "가평 여행", "photo_id": [str(p1), str(p2)]},
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/event/", response.headers["Location"])
        events = repository.list_events(self.conn)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["instance_count"], 2)

    def test_bulk_add_with_no_photos_selected_is_a_no_op(self):
        response = self.client.post("/events/bulk-add", data={"event_name": "여행", "photo_id": []})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(repository.list_events(self.conn), [])

    def test_bulk_add_with_no_name_is_a_no_op(self):
        photo_id = self._photo()

        self.client.post("/events/bulk-add", data={"event_name": "", "photo_id": [str(photo_id)]})

        self.assertEqual(repository.list_events(self.conn), [])

    def test_bulk_add_reuses_an_existing_event(self):
        p1, p2 = self._photo(), self._photo()
        self.client.post("/events/bulk-add", data={"event_name": "여행", "photo_id": [str(p1)]})

        self.client.post("/events/bulk-add", data={"event_name": "여행", "photo_id": [str(p2)]})

        events = repository.list_events(self.conn)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["instance_count"], 2)

    def test_event_detail_page_renders(self):
        photo_id = self._photo()
        with self.conn:
            event_id = repository.create_event(self.conn, "가평 여행")
            repository.add_photos_to_event(self.conn, event_id, [photo_id])

        response = self.client.get(f"/event/{event_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("가평 여행".encode(), response.data)

    def test_unknown_event_is_404(self):
        self.assertEqual(self.client.get("/event/9999").status_code, 404)

    def test_rename_route(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "old")

        response = self.client.post(f"/event/{event_id}/rename", data={"name": "new"})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(repository.event_detail(self.conn, event_id)["event"]["name"], "new")

    def test_description_route(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")

        self.client.post(f"/event/{event_id}/description", data={"description": "좋았다"})

        self.assertEqual(repository.event_detail(self.conn, event_id)["event"]["description"], "좋았다")

    def test_remove_photo_route(self):
        photo_id = self._photo()
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.add_photos_to_event(self.conn, event_id, [photo_id])

        self.client.post(f"/event/{event_id}/remove-photo", data={"photo_id": str(photo_id)})

        self.assertEqual(repository.event_detail(self.conn, event_id)["photos"], [])

    def test_delete_route(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")

        response = self.client.post(f"/event/{event_id}/delete")

        self.assertEqual(response.status_code, 302)
        self.assertIsNone(repository.event_detail(self.conn, event_id))

    def test_cover_route_uses_the_first_photo(self):
        photo_id = self._photo()
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.add_photos_to_event(self.conn, event_id, [photo_id])

        response = self.client.get(f"/event/{event_id}/cover")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")

    def test_cover_route_404s_for_an_empty_event(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "빈이벤트")

        self.assertEqual(self.client.get(f"/event/{event_id}/cover").status_code, 404)

    def test_autobio_exclude_route_persists(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")

        response = self.client.post(f"/event/{event_id}/autobio-exclude", data={"excluded": "1"})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            repository.event_detail(self.conn, event_id)["event"]["excluded_from_autobio"], 1
        )

    def test_autobio_exclude_route_can_be_unset(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.set_event_autobio_excluded(self.conn, event_id, True)

        # An unchecked checkbox submits no "excluded" field at all — the
        # route must treat that as false, not require an explicit "0".
        self.client.post(f"/event/{event_id}/autobio-exclude", data={})

        self.assertEqual(
            repository.event_detail(self.conn, event_id)["event"]["excluded_from_autobio"], 0
        )

    def test_excluded_badge_shows_on_the_groups_index(self):
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.set_event_autobio_excluded(self.conn, event_id, True)

        response = self.client.get("/events")

        # "일기/자서전 제외됨" (part of groups_page.excluded_badge) —
        # ui_language defaults to Korean.
        self.assertIn("일기/자서전 제외됨".encode(), response.data)


class AutobioExclusionAffectsPhotosForDateTests(EventTestCase):
    """The actual point of the feature: photos_for_date(), which feeds the
    Autobio/Diary generation prompt, must drop photos tagged with an
    excluded event — not just hide the event's name."""

    def test_a_photo_only_in_an_excluded_event_is_dropped_entirely(self):
        photo_id = self._photo()
        with self.conn:
            event_id = repository.create_event(self.conn, "비공개 모임")
            repository.add_photos_to_event(self.conn, event_id, [photo_id])
            repository.set_event_autobio_excluded(self.conn, event_id, True)

        photos = repository.photos_for_date(self.conn, "2026-04-11")

        self.assertEqual(photos, [])

    def test_a_photo_in_a_non_excluded_event_still_appears(self):
        photo_id = self._photo()
        with self.conn:
            event_id = repository.create_event(self.conn, "여행")
            repository.add_photos_to_event(self.conn, event_id, [photo_id])

        photos = repository.photos_for_date(self.conn, "2026-04-11")

        self.assertEqual(len(photos), 1)
        self.assertEqual(photos[0]["events"], ["여행"])

    def test_a_photo_in_both_an_excluded_and_a_normal_event_is_still_dropped(self):
        # Excluding wins over including — the whole point is "never let
        # this photo show up in generated text," regardless of its other tags.
        photo_id = self._photo()
        with self.conn:
            normal_id = repository.create_event(self.conn, "여행")
            excluded_id = repository.create_event(self.conn, "비공개 모임")
            repository.add_photos_to_event(self.conn, normal_id, [photo_id])
            repository.add_photos_to_event(self.conn, excluded_id, [photo_id])
            repository.set_event_autobio_excluded(self.conn, excluded_id, True)

        photos = repository.photos_for_date(self.conn, "2026-04-11")

        self.assertEqual(photos, [])

    def test_a_photo_with_no_event_at_all_is_unaffected(self):
        self._photo()

        photos = repository.photos_for_date(self.conn, "2026-04-11")

        self.assertEqual(len(photos), 1)


if __name__ == "__main__":
    unittest.main()
