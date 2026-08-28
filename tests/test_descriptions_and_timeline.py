"""Descriptions on clusters/photos, and the "photo by time" timeline view —
both new, not in the original spec.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402

DIMS = 512


def _embedding(person: int) -> list[float]:
    vector = [0.0] * DIMS
    vector[person % DIMS] = 1.0
    return vector


class DescTimelineTestCase(unittest.TestCase):
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

    def _photo(self, taken_at=None) -> int:
        from PIL import Image

        self._n += 1
        name = f"p{self._n}.jpg"
        path = self.tmp / name
        Image.new("RGB", (200, 200), color=(90, 90, 90)).save(path, "JPEG")
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at=taken_at or f"2026-04-{10 + self._n:02d}T10:00:00",
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )

    def _face_cluster(self, photo_id, person=0) -> int:
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(person),
            )
        return cluster_id

    def _place(self, photo_id, name=None):
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=37.5, lng=127.0)
            repository.set_photo_location(self.conn, photo_id=photo_id, location_cluster_id=cluster_id)
            if name:
                repository.name_cluster(self.conn, "place", cluster_id, name)
        return cluster_id


class ClusterDescriptionTests(DescTimelineTestCase):
    def test_sets_description_on_a_face_cluster(self):
        cluster_id = self._face_cluster(self._photo())

        with self.conn:
            repository.set_cluster_description(self.conn, "face", cluster_id, "우리 엄마")

        self.assertEqual(repository.cluster_row(self.conn, "face", cluster_id)["description"], "우리 엄마")

    def test_sets_description_on_a_place_cluster(self):
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=37.5, lng=127.0)
            repository.set_cluster_description(self.conn, "place", cluster_id, "자주 가는 곳")

        self.assertEqual(repository.cluster_row(self.conn, "place", cluster_id)["description"], "자주 가는 곳")

    def test_blank_description_is_stored_as_null(self):
        cluster_id = self._face_cluster(self._photo())

        with self.conn:
            repository.set_cluster_description(self.conn, "face", cluster_id, "   ")

        self.assertIsNone(repository.cluster_row(self.conn, "face", cluster_id)["description"])

    def test_route_saves_and_redirects(self):
        cluster_id = self._face_cluster(self._photo())

        response = self.client.post(
            f"/cluster/face/{cluster_id}/description", data={"description": "우리 엄마"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(repository.cluster_row(self.conn, "face", cluster_id)["description"], "우리 엄마")

    def test_description_appears_on_the_detail_page(self):
        cluster_id = self._face_cluster(self._photo())
        with self.conn:
            repository.set_cluster_description(self.conn, "face", cluster_id, "우리 엄마 설명")

        response = self.client.get(f"/cluster/face/{cluster_id}")

        self.assertIn("우리 엄마 설명".encode(), response.data)


class PhotoDescriptionTests(DescTimelineTestCase):
    def test_sets_and_reads_back(self):
        photo_id = self._photo()

        with self.conn:
            repository.set_photo_description(self.conn, photo_id, "좋은 하루였다")

        self.assertEqual(
            repository.photo_detail(self.conn, photo_id)["photo"]["description"], "좋은 하루였다"
        )

    def test_route_saves_and_redirects(self):
        photo_id = self._photo()

        response = self.client.post(
            f"/photo/{photo_id}/description", data={"description": "메모"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(repository.photo_detail(self.conn, photo_id)["photo"]["description"], "메모")


class TimelineTests(DescTimelineTestCase):
    def test_newest_photo_first(self):
        early = self._photo(taken_at="2020-01-01T00:00:00")
        late = self._photo(taken_at="2026-01-01T00:00:00")

        photos = repository.list_all_photos(self.conn)

        self.assertEqual([p["photo_id"] for p in photos], [late, early])

    def test_pagination_limit_and_offset(self):
        ids = [self._photo(taken_at=f"2026-01-{i:02d}T00:00:00") for i in range(1, 6)]

        page = repository.list_all_photos(self.conn, limit=2, offset=2)

        # Newest-first ordering: ids[4] is newest, ids[0] oldest.
        expected = list(reversed(ids))[2:4]
        self.assertEqual([p["photo_id"] for p in page], expected)

    def test_total_count(self):
        for _ in range(3):
            self._photo()
        self.assertEqual(repository.total_photo_count(self.conn), 3)

    def test_includes_photos_with_no_faces_or_location(self):
        # The whole point of "by time" is to see everything, not just
        # sorted photos.
        self._photo()
        self.assertEqual(len(repository.list_all_photos(self.conn)), 1)

    def test_route_renders(self):
        self._photo()
        response = self.client.get("/timeline")
        self.assertEqual(response.status_code, 200)

    def test_route_paginates(self):
        for _ in range(3):
            self._photo()

        response = self.client.get("/timeline?page=1")

        self.assertIn("사진 3장".encode(), response.data)  # ui_language defaults to Korean

    def test_checkboxes_present_for_bulk_event_assignment(self):
        self._photo()
        response = self.client.get("/timeline")
        self.assertIn(b'name="photo_id"', response.data)
        self.assertIn(b"event_name", response.data)

    def test_list_all_photos_includes_place_name_when_labeled(self):
        photo_id = self._photo()
        self._place(photo_id, name="Seoul, South Korea")

        photos = repository.list_all_photos(self.conn)

        self.assertEqual(photos[0]["place_name"], "Seoul, South Korea")

    def test_list_all_photos_place_name_is_none_when_unlabeled(self):
        photo_id = self._photo()

        photos = repository.list_all_photos(self.conn)

        self.assertIsNone(photos[0]["place_name"])


class TimelineGroupingTests(DescTimelineTestCase):
    """Google Photos groups its own timeline by day, with a header showing
    the date and (when known) a place — this is the same shape, built from
    repository.list_all_photos()'s newest-first rows."""

    def test_photos_on_different_days_get_separate_groups(self):
        self._photo(taken_at="2026-08-15T09:00:00")
        self._photo(taken_at="2026-08-18T09:00:00")

        groups = web._group_photos_by_date(repository.list_all_photos(self.conn))

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]["photos"]), 1)
        self.assertEqual(len(groups[1]["photos"]), 1)

    def test_photos_on_the_same_day_share_a_group(self):
        self._photo(taken_at="2026-08-18T09:00:00")
        self._photo(taken_at="2026-08-18T18:30:00")

        groups = web._group_photos_by_date(repository.list_all_photos(self.conn))

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["photos"]), 2)

    def test_group_label_omits_year_for_the_current_year(self):
        from datetime import date

        this_year = date.today().year
        groups = web._group_photos_by_date(
            [{"photo_id": 1, "taken_at": f"{this_year}-08-18T09:00:00", "place_name": None}]
        )

        self.assertNotIn(str(this_year), groups[0]["date_label"])

    def test_group_label_includes_year_for_a_past_year(self):
        groups = web._group_photos_by_date(
            [{"photo_id": 1, "taken_at": "2020-08-18T09:00:00", "place_name": None}]
        )

        self.assertIn("2020", groups[0]["date_label"])

    def test_group_place_name_shown_when_every_photo_in_the_day_agrees(self):
        p1 = self._photo(taken_at="2026-08-18T09:00:00")
        p2 = self._photo(taken_at="2026-08-18T18:00:00")
        self._place(p1, name="Seoul, South Korea")
        self._place(p2, name="Seoul, South Korea")

        groups = web._group_photos_by_date(repository.list_all_photos(self.conn))

        self.assertEqual(groups[0]["place_name"], "Seoul, South Korea")

    def test_group_place_name_omitted_when_the_day_spans_multiple_places(self):
        p1 = self._photo(taken_at="2026-08-18T09:00:00")
        p2 = self._photo(taken_at="2026-08-18T18:00:00")
        self._place(p1, name="Seoul, South Korea")
        self._place(p2, name="Busan, South Korea")

        groups = web._group_photos_by_date(repository.list_all_photos(self.conn))

        self.assertIsNone(groups[0]["place_name"])

    def test_route_renders_date_headers_and_group_select_all(self):
        photo_id = self._photo(taken_at="2026-08-18T09:00:00")
        self._place(photo_id, name="Seoul, South Korea")

        response = self.client.get("/timeline")

        self.assertIn(b"Tue, Aug 18", response.data)
        self.assertIn(b"Seoul, South Korea", response.data)
        self.assertIn(b"tl-group-select", response.data)
        self.assertIn(b'data-group="0"', response.data)


if __name__ == "__main__":
    unittest.main()
