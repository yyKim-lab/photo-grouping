"""Labeling UI tests, driven through Flask's test client against a real
temporary database. No browser, no network.

Image routes are exercised too, since they do real cropping work off disk —
a broken bounding-box conversion would otherwise only show up as a broken
image in the browser.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402


def _make_photo_file(path: Path, size=(800, 600)) -> None:
    from PIL import Image

    Image.new("RGB", size, color=(120, 90, 70)).save(path, "JPEG")


class WebTestCase(unittest.TestCase):
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

    # -- fixtures ----------------------------------------------------

    def _add_photo(self, name="a.jpg", **kwargs) -> int:
        photo_path = self.tmp / name
        _make_photo_file(photo_path)
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=kwargs.pop("picker_media_id", f"pmi-{name}"),
                taken_at=kwargs.pop("taken_at", "2026-04-12T10:00:00"),
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(photo_path),
                **kwargs,
            )

    def _add_face_cluster(self, photo_ids, box=None) -> int:
        box = box or {"x": 0.25, "y": 0.25, "width": 0.3, "height": 0.3}
        with self.conn:
            cluster_id = repository.insert_face_cluster(
                self.conn, representative_photo_id=photo_ids[0]
            )
            for photo_id in photo_ids:
                repository.insert_face_instance(
                    self.conn,
                    photo_id=photo_id,
                    face_cluster_id=cluster_id,
                    bounding_box=box,
                    embedding=[0.1] * 128,
                )
        return cluster_id

    def _add_place_cluster(self, photo_ids, lat=37.5, lng=127.0) -> int:
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=lat, lng=lng)
            for photo_id in photo_ids:
                repository.set_photo_location(
                    self.conn, photo_id=photo_id, location_cluster_id=cluster_id
                )
        return cluster_id


class IndexTests(WebTestCase):
    def test_empty_library_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("아직 이름이 붙은 사람이 없습니다".encode(), response.data)  # ui_language defaults to Korean

    def test_named_cluster_appears_as_an_album(self):
        cluster_id = self._add_face_cluster([self._add_photo()])
        with self.conn:
            repository.name_cluster(self.conn, "face", cluster_id, "Youngju")

        response = self.client.get("/")
        self.assertIn(b"Youngju", response.data)

    def test_not_sure_cluster_is_badged_in_uncategorized(self):
        cluster_id = self._add_face_cluster([self._add_photo()])
        with self.conn:
            repository.mark_cluster_not_sure(self.conn, "face", cluster_id)

        response = self.client.get("/")
        self.assertIn("확실하지 않음".encode(), response.data)  # ui_language defaults to Korean


class QueueTests(WebTestCase):
    def test_shows_a_face_card_with_the_person_prompt(self):
        self._add_face_cluster([self._add_photo()])

        response = self.client.get("/queue")
        self.assertEqual(response.status_code, 200)
        self.assertIn("이 사람을 아십니까?".encode(), response.data)  # ui_language defaults to Korean

    def test_shows_a_place_card_with_the_place_prompt(self):
        self._add_place_cluster([self._add_photo()])

        response = self.client.get("/queue")
        self.assertIn("이 장소를 아십니까?".encode(), response.data)  # ui_language defaults to Korean

    def test_empty_state_when_nothing_needs_labeling(self):
        response = self.client.get("/queue")
        self.assertIn("더 이상 라벨링할 항목이 없습니다".encode(), response.data)  # ui_language defaults to Korean

    def test_largest_cluster_is_offered_first(self):
        # Naming the person in 3 photos resolves more of the library per
        # decision than naming a one-off, so the queue front-loads it.
        small = self._add_face_cluster([self._add_photo("small.jpg")])
        big = self._add_face_cluster(
            [self._add_photo(f"big{i}.jpg") for i in range(3)]
        )

        queue = repository.load_labeling_queue(self.conn)
        self.assertEqual(queue[0]["id"], big)
        self.assertEqual(queue[0]["instance_count"], 3)
        self.assertEqual(queue[-1]["id"], small)

    def test_naming_a_cluster_removes_it_from_the_queue(self):
        cluster_id = self._add_face_cluster([self._add_photo()])

        self.client.post(f"/cluster/face/{cluster_id}/name", data={"name": "Youngju"})

        self.assertEqual(repository.load_labeling_queue(self.conn), [])
        row = repository.cluster_row(self.conn, "face", cluster_id)
        self.assertEqual(row["status"], "named")
        self.assertEqual(row["name"], "Youngju")

    def test_empty_name_is_rejected(self):
        # A "named" cluster with a blank label would disappear from both
        # Albums and Uncategorized.
        cluster_id = self._add_face_cluster([self._add_photo()])

        self.client.post(f"/cluster/face/{cluster_id}/name", data={"name": "   "})

        self.assertEqual(repository.cluster_row(self.conn, "face", cluster_id)["status"], "unlabeled")

    def test_not_sure_removes_from_queue_until_the_cluster_grows(self):
        photo_ids = [self._add_photo(f"p{i}.jpg") for i in range(2)]
        cluster_id = self._add_face_cluster(photo_ids)

        self.client.post(f"/cluster/face/{cluster_id}/not-sure")
        self.assertEqual(repository.load_labeling_queue(self.conn), [])

        # Four more photos: still short of the +5 regrow threshold... no,
        # exactly 4 is short of 5.
        self._add_faces_to_cluster(cluster_id, 4)
        self.assertEqual(repository.load_labeling_queue(self.conn), [])

        # The fifth brings it back.
        self._add_faces_to_cluster(cluster_id, 1)
        queue = repository.load_labeling_queue(self.conn)
        self.assertEqual([item["id"] for item in queue], [cluster_id])
        self.assertEqual(queue[0]["status"], "not_sure")

    _extra_counter = 0

    def _add_faces_to_cluster(self, cluster_id, n):
        with self.conn:
            for _ in range(n):
                QueueTests._extra_counter += 1
                photo_id = self._add_photo(f"extra_{QueueTests._extra_counter}.jpg")
                repository.insert_face_instance(
                    self.conn,
                    photo_id=photo_id,
                    face_cluster_id=cluster_id,
                    bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                    embedding=[0.1] * 128,
                )

    def test_place_regrow_rule_works_too(self):
        # location_cluster only gained last_shown_at/instance_count_at_last_shown
        # in migration 0003 — §2's pseudocode omitted them, but §4.2 applies
        # the same rule to places.
        cluster_id = self._add_place_cluster([self._add_photo("p0.jpg")])

        self.client.post(f"/cluster/place/{cluster_id}/not-sure")
        self.assertEqual(repository.load_labeling_queue(self.conn), [])

        with self.conn:
            for i in range(5):
                repository.set_photo_location(
                    self.conn,
                    photo_id=self._add_photo(f"more{i}.jpg"),
                    location_cluster_id=cluster_id,
                )

        queue = repository.load_labeling_queue(self.conn)
        self.assertEqual([item["id"] for item in queue], [cluster_id])


class ClusterDetailTests(WebTestCase):
    def test_default_view_shows_photos_not_crops(self):
        # Photos is the default: browsing your own album should show the
        # actual pictures. Crops are for judging "is this the same person?"
        # during split, reached via the explicit Faces toggle.
        cluster_id = self._add_face_cluster(
            [self._add_photo(f"p{i}.jpg") for i in range(3)]
        )

        response = self.client.get(f"/cluster/face/{cluster_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.count(b"/thumb"), 3)
        self.assertEqual(response.data.count(b"/crop"), 0)

    def test_faces_view_shows_every_member_not_just_the_representative(self):
        cluster_id = self._add_face_cluster(
            [self._add_photo(f"p{i}.jpg") for i in range(3)]
        )

        response = self.client.get(f"/cluster/face/{cluster_id}?view=faces")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.count(b"/crop"), 3)

    def test_renaming_from_detail_view_works_on_a_named_cluster(self):
        # §4.3: corrections are available from any entry point, including
        # an already-named album.
        cluster_id = self._add_face_cluster([self._add_photo()])
        with self.conn:
            repository.name_cluster(self.conn, "face", cluster_id, "Wrong Name")

        self.client.post(f"/cluster/face/{cluster_id}/name", data={"name": "Right Name"})

        self.assertEqual(repository.cluster_row(self.conn, "face", cluster_id)["name"], "Right Name")

    def test_unknown_cluster_is_404(self):
        self.assertEqual(self.client.get("/cluster/face/9999").status_code, 404)

    def test_unknown_kind_is_404(self):
        self.assertEqual(self.client.get("/cluster/banana/1").status_code, 404)


class ImageRouteTests(WebTestCase):
    def test_photo_thumb_returns_a_jpeg(self):
        photo_id = self._add_photo()

        response = self.client.get(f"/photo/{photo_id}/thumb")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertTrue(response.data.startswith(b"\xff\xd8"))  # JPEG magic

    def test_face_crop_uses_the_fractional_bounding_box(self):
        from PIL import Image

        photo_id = self._add_photo(name="big.jpg")
        cluster_id = self._add_face_cluster(
            [photo_id], box={"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}
        )
        face_id = self.conn.execute(
            "SELECT id FROM face_instance WHERE face_cluster_id = ?", (cluster_id,)
        ).fetchone()["id"]

        response = self.client.get(f"/face/{face_id}/crop")

        self.assertEqual(response.status_code, 200)
        crop = Image.open(__import__("io").BytesIO(response.data))
        # The source is 800x600; a half-size box plus 40% padding on each
        # side is ~720x540, capped by the 400px thumbnail limit. What
        # matters is that it's a real crop, not the whole frame.
        self.assertLessEqual(max(crop.size), web.app.config["THUMBNAIL_MAX_PX"])
        self.assertGreater(min(crop.size), 0)

    def test_missing_original_file_is_404_not_a_crash(self):
        photo_id = self._add_photo()
        (self.tmp / "a.jpg").unlink()

        self.assertEqual(self.client.get(f"/photo/{photo_id}/thumb").status_code, 404)


if __name__ == "__main__":
    unittest.main()
