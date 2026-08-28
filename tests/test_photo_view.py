"""Person-album photo browsing (vs. the crop-based split view) and the
full-screen viewer's backing route.
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


class PhotoViewTestCase(unittest.TestCase):
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

    def _photo(self, size=(300, 200)) -> int:
        from PIL import Image

        self._n += 1
        name = f"p{self._n}.jpg"
        path = self.tmp / name
        Image.new("RGB", size, color=(90, 90, 90)).save(path, "JPEG")
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at="2026-04-12T10:00:00",
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )

    def _face_cluster(self, photo_ids, person=0) -> int:
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            for photo_id in photo_ids:
                repository.insert_face_instance(
                    self.conn,
                    photo_id=photo_id,
                    face_cluster_id=cluster_id,
                    bounding_box={"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3},
                    embedding=_embedding(person),
                )
        return cluster_id


class PhotosInFaceClusterTests(PhotoViewTestCase):
    def test_returns_one_entry_per_distinct_photo(self):
        photo_ids = [self._photo() for _ in range(3)]
        cluster_id = self._face_cluster(photo_ids)

        photos = repository.photos_in_face_cluster(self.conn, cluster_id)

        self.assertEqual({p["photo_id"] for p in photos}, set(photo_ids))
        self.assertEqual(len(photos), 3)

    def test_a_photo_with_two_faces_of_the_same_person_appears_once(self):
        # Rare (a mirror, a photo of a photo) but should never double the
        # photo in the album.
        photo_id = self._photo()
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            for _ in range(2):
                repository.insert_face_instance(
                    self.conn,
                    photo_id=photo_id,
                    face_cluster_id=cluster_id,
                    bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                    embedding=_embedding(0),
                )

        photos = repository.photos_in_face_cluster(self.conn, cluster_id)

        self.assertEqual(len(photos), 1)

    def test_ordered_by_taken_at(self):
        with self.conn:
            early = repository.insert_photo(
                self.conn, picker_media_id="e", taken_at="2026-01-01T00:00:00",
                original_filename="e.jpg", original_storage_backend="local",
                original_storage_path=str(self.tmp / "e.jpg"),
            )
            late = repository.insert_photo(
                self.conn, picker_media_id="l", taken_at="2026-06-01T00:00:00",
                original_filename="l.jpg", original_storage_backend="local",
                original_storage_path=str(self.tmp / "l.jpg"),
            )
        cluster_id = self._face_cluster([late, early])

        photos = repository.photos_in_face_cluster(self.conn, cluster_id)

        self.assertEqual([p["photo_id"] for p in photos], [early, late])


class DetailViewToggleTests(PhotoViewTestCase):
    def test_photos_is_the_default_view(self):
        cluster_id = self._face_cluster([self._photo()])

        response = self.client.get(f"/cluster/face/{cluster_id}")

        self.assertIn("이 그룹의 사진".encode(), response.data)  # ui_language defaults to Korean

    def test_explicit_faces_view_shows_the_split_form(self):
        cluster_id = self._face_cluster([self._photo(), self._photo()])

        response = self.client.get(f"/cluster/face/{cluster_id}?view=faces")

        self.assertIn("이 그룹의 얼굴".encode(), response.data)  # ui_language defaults to Korean
        self.assertIn(b"face_instance_id", response.data)

    def test_photo_view_has_no_split_checkboxes(self):
        cluster_id = self._face_cluster([self._photo()])

        response = self.client.get(f"/cluster/face/{cluster_id}")

        self.assertNotIn(b"face_instance_id", response.data)

    def test_invalid_view_param_falls_back_to_photos(self):
        cluster_id = self._face_cluster([self._photo()])

        response = self.client.get(f"/cluster/face/{cluster_id}?view=nonsense")

        self.assertIn("이 그룹의 사진".encode(), response.data)  # ui_language defaults to Korean

    def test_toggle_links_are_present_and_point_at_both_views(self):
        cluster_id = self._face_cluster([self._photo()])

        response = self.client.get(f"/cluster/face/{cluster_id}")

        self.assertIn(f"/cluster/face/{cluster_id}?view=photos".encode(), response.data)
        self.assertIn(f"/cluster/face/{cluster_id}?view=faces".encode(), response.data)

    def test_place_clusters_have_no_toggle(self):
        # Places only ever have one kind of "photos in this cluster" view —
        # there's no crop-based split equivalent for a location.
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=37.5, lng=127.0)
            repository.set_photo_location(
                self.conn, photo_id=self._photo(), location_cluster_id=cluster_id
            )

        response = self.client.get(f"/cluster/place/{cluster_id}")

        self.assertNotIn(b"for splitting", response.data)


class LightboxMarkupTests(PhotoViewTestCase):
    def test_photo_grid_images_carry_full_size_and_caption_data(self):
        cluster_id = self._face_cluster([self._photo()])

        response = self.client.get(f"/cluster/face/{cluster_id}")
        body = response.data.decode()

        self.assertIn("data-full=", body)
        self.assertIn("/full", body)
        self.assertIn("onclick=\"openLightbox(this)\"", body)

    def test_lightbox_overlay_markup_is_present(self):
        cluster_id = self._face_cluster([self._photo()])

        response = self.client.get(f"/cluster/face/{cluster_id}")

        self.assertIn(b'id="lightbox"', response.data)
        self.assertIn(b"lightboxStep", response.data)


class PhotoFullRouteTests(PhotoViewTestCase):
    def test_returns_a_larger_image_than_the_thumbnail(self):
        # Big enough that the 400px thumbnail cap and the 1800px full cap
        # produce visibly different sizes.
        photo_id = self._photo(size=(3000, 2000))

        thumb = self.client.get(f"/photo/{photo_id}/thumb")
        full = self.client.get(f"/photo/{photo_id}/full")

        from io import BytesIO

        from PIL import Image

        thumb_img = Image.open(BytesIO(thumb.data))
        full_img = Image.open(BytesIO(full.data))

        self.assertLessEqual(max(thumb_img.size), 400)
        self.assertGreater(max(full_img.size), max(thumb_img.size))
        self.assertLessEqual(max(full_img.size), 1800)

    def test_is_still_capped_not_the_raw_original(self):
        # Re-encoded and capped, not a passthrough of the original file —
        # matters for response size on a real (multi-MB) phone photo.
        photo_id = self._photo(size=(4000, 3000))

        response = self.client.get(f"/photo/{photo_id}/full")

        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(response.data))
        self.assertLessEqual(max(img.size), 1800)

    def test_missing_file_is_404(self):
        photo_id = self._photo()
        (self.tmp / "p1.jpg").unlink()

        self.assertEqual(self.client.get(f"/photo/{photo_id}/full").status_code, 404)


if __name__ == "__main__":
    unittest.main()
