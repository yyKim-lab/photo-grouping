"""Hiding clusters — strangers caught in the background of photos.

Exclusion is orthogonal to labeling status (see migration 0004), so these
check both that hidden clusters disappear from every surface that offers
work, and that their labeling state survives a hide/restore round trip.
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


class ExcludeTestCase(unittest.TestCase):
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

    def _face_cluster(self, n_faces=1, person=0, name=None) -> int:
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
            if name:
                repository.name_cluster(self.conn, "face", cluster_id, name)
        return cluster_id

    def _place_cluster(self, n_photos=1, name=None) -> int:
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=37.5, lng=127.0)
            for _ in range(n_photos):
                repository.set_photo_location(
                    self.conn, photo_id=self._photo(), location_cluster_id=cluster_id
                )
            if name:
                repository.name_cluster(self.conn, "place", cluster_id, name)
        return cluster_id


class HidingRemovesFromSurfacesTests(ExcludeTestCase):
    def test_hidden_cluster_leaves_the_labeling_queue(self):
        stranger = self._face_cluster(2)
        self._face_cluster(3, person=1)

        self.assertEqual(len(repository.load_labeling_queue(self.conn)), 2)
        with self.conn:
            repository.exclude_cluster(self.conn, "face", stranger)

        queue = repository.load_labeling_queue(self.conn)
        self.assertEqual([i["id"] for i in queue], [i["id"] for i in queue if i["id"] != stranger])
        self.assertNotIn(stranger, [i["id"] for i in queue])

    def test_hidden_cluster_leaves_uncategorized(self):
        stranger = self._face_cluster(2)

        with self.conn:
            repository.exclude_cluster(self.conn, "face", stranger)

        self.assertEqual(repository.load_uncategorized(self.conn)["people"], [])

    def test_hidden_named_cluster_leaves_albums(self):
        # A named cluster can be hidden too — someone you can identify but
        # don't want in your albums.
        cluster_id = self._face_cluster(2, name="Stranger")

        with self.conn:
            repository.exclude_cluster(self.conn, "face", cluster_id)

        self.assertEqual(repository.load_albums(self.conn)["people"], [])

    def test_hidden_cluster_is_not_offered_as_a_merge_candidate(self):
        target = self._face_cluster(2, person=0)
        stranger = self._face_cluster(1, person=0)  # identical embedding

        self.assertIn(stranger, [c["id"] for c in repository.similar_face_clusters(self.conn, target)])
        with self.conn:
            repository.exclude_cluster(self.conn, "face", stranger)

        self.assertNotIn(
            stranger, [c["id"] for c in repository.similar_face_clusters(self.conn, target)]
        )

    def test_hidden_place_leaves_the_queue(self):
        place = self._place_cluster(2)

        with self.conn:
            repository.exclude_cluster(self.conn, "place", place)

        self.assertEqual(repository.load_labeling_queue(self.conn), [])


class RestoreTests(ExcludeTestCase):
    def test_restoring_brings_it_back(self):
        cluster_id = self._face_cluster(2)
        with self.conn:
            repository.exclude_cluster(self.conn, "face", cluster_id)
            repository.restore_cluster(self.conn, "face", cluster_id)

        self.assertEqual(len(repository.load_labeling_queue(self.conn)), 1)

    def test_labeling_state_survives_a_hide_restore_round_trip(self):
        # Exclusion is orthogonal to status, so a "not sure" cluster comes
        # back as "not sure", not reset to unlabeled.
        cluster_id = self._face_cluster(2)
        with self.conn:
            repository.mark_cluster_not_sure(self.conn, "face", cluster_id)
            repository.exclude_cluster(self.conn, "face", cluster_id)
            repository.restore_cluster(self.conn, "face", cluster_id)

        self.assertEqual(repository.cluster_row(self.conn, "face", cluster_id)["status"], "not_sure")

    def test_nothing_is_deleted_by_hiding(self):
        cluster_id = self._face_cluster(3)
        with self.conn:
            repository.exclude_cluster(self.conn, "face", cluster_id)

        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) c FROM face_instance").fetchone()["c"], 3
        )
        self.assertIsNotNone(repository.cluster_row(self.conn, "face", cluster_id))

    def test_load_excluded_lists_hidden_clusters(self):
        hidden = self._face_cluster(2, name="Passerby")
        self._face_cluster(1)

        with self.conn:
            repository.exclude_cluster(self.conn, "face", hidden)

        excluded = repository.load_excluded(self.conn)
        self.assertEqual([c["id"] for c in excluded["people"]], [hidden])
        self.assertEqual(excluded["people"][0]["name"], "Passerby")

    def test_excluded_count_covers_both_kinds(self):
        with self.conn:
            repository.exclude_cluster(self.conn, "face", self._face_cluster(1))
            repository.exclude_cluster(self.conn, "place", self._place_cluster(1))

        self.assertEqual(repository.excluded_count(self.conn), 2)


class RouteTests(ExcludeTestCase):
    def test_queue_offers_the_exclude_button(self):
        self._face_cluster(2)

        response = self.client.get("/queue")

        self.assertIn("모르는 사람".encode(), response.data)  # ui_language defaults to Korean

    def test_place_card_uses_place_wording(self):
        self._place_cluster(2)

        response = self.client.get("/queue")

        self.assertIn("관심 없는 장소".encode(), response.data)  # ui_language defaults to Korean

    def test_exclude_route_hides_and_returns_to_the_queue(self):
        cluster_id = self._face_cluster(2)
        self._face_cluster(1, person=1)

        response = self.client.post(f"/cluster/face/{cluster_id}/exclude")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/queue", response.headers["Location"])
        self.assertNotIn(
            cluster_id, [i["id"] for i in repository.load_labeling_queue(self.conn)]
        )

    def test_restore_route_brings_it_back(self):
        cluster_id = self._face_cluster(2)
        self.client.post(f"/cluster/face/{cluster_id}/exclude")

        self.client.post(f"/cluster/face/{cluster_id}/restore")

        self.assertEqual(len(repository.load_labeling_queue(self.conn)), 1)

    def test_hidden_page_lists_and_offers_restore(self):
        cluster_id = self._face_cluster(2, name="Passerby")
        self.client.post(f"/cluster/face/{cluster_id}/exclude")

        response = self.client.get("/excluded")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Passerby", response.data)
        self.assertIn(f"/cluster/face/{cluster_id}/restore".encode(), response.data)

    def test_hidden_page_empty_state(self):
        response = self.client.get("/excluded")
        self.assertIn("숨겨진 항목이 없습니다".encode(), response.data)  # ui_language defaults to Korean

    def test_albums_page_links_to_hidden_when_there_are_any(self):
        cluster_id = self._face_cluster(2)
        self.client.post(f"/cluster/face/{cluster_id}/exclude")

        response = self.client.get("/")

        self.assertIn("숨김 1건".encode(), response.data)  # ui_language defaults to Korean

    def test_cluster_detail_offers_hide_then_unhide(self):
        cluster_id = self._face_cluster(2)

        self.assertIn("이 사람 숨기기".encode(), self.client.get(f"/cluster/face/{cluster_id}").data)

        self.client.post(f"/cluster/face/{cluster_id}/exclude")

        self.assertIn("숨김 해제".encode(), self.client.get(f"/cluster/face/{cluster_id}").data)

    def test_unknown_kind_is_rejected(self):
        self.assertEqual(self.client.post("/cluster/banana/1/exclude").status_code, 404)


if __name__ == "__main__":
    unittest.main()
