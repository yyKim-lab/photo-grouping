"""Per-photo metadata edits (§4.4): reassigning a face, marking a false
positive, manually adding a missed face, and overriding date/location.
"""

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, location_clustering, repository, web  # noqa: E402

DIMS = 512


def _embedding(person: int) -> list[float]:
    vector = [0.0] * DIMS
    vector[person % DIMS] = 1.0
    return vector


class PhotoEditTestCase(unittest.TestCase):
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

    def _photo(self, size=(400, 300)) -> int:
        from PIL import Image

        self._n += 1
        name = f"p{self._n}.jpg"
        path = self.tmp / name
        Image.new("RGB", size, color=(80, 80, 80)).save(path, "JPEG")
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at="2026-04-12T10:00:00",
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )

    def _face(self, photo_id, person=0, cluster_name=None) -> tuple[int, int]:
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            face_id = repository.insert_face_instance(
                self.conn,
                photo_id=photo_id,
                face_cluster_id=cluster_id,
                bounding_box={"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3},
                embedding=_embedding(person),
            )
            if cluster_name:
                repository.name_cluster(self.conn, "face", cluster_id, cluster_name)
        return face_id, cluster_id


class PhotoDetailTests(PhotoEditTestCase):
    def test_returns_none_for_unknown_photo(self):
        self.assertIsNone(repository.photo_detail(self.conn, 9999))

    def test_lists_faces_with_their_cluster_names(self):
        photo_id = self._photo()
        self._face(photo_id, cluster_name="엄마")

        detail = repository.photo_detail(self.conn, photo_id)

        self.assertEqual(len(detail["faces"]), 1)
        self.assertEqual(detail["faces"][0]["cluster_name"], "엄마")

    def test_false_positive_faces_are_excluded(self):
        photo_id = self._photo()
        face_id, _ = self._face(photo_id)
        with self.conn:
            repository.mark_face_false_positive(self.conn, face_id)

        detail = repository.photo_detail(self.conn, photo_id)

        self.assertEqual(detail["faces"], [])

    def test_includes_location_when_set(self):
        photo_id = self._photo()
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=37.5, lng=127.0)
            repository.name_cluster(self.conn, "place", cluster_id, "우리집")
            repository.set_photo_location(self.conn, photo_id=photo_id, location_cluster_id=cluster_id)

        detail = repository.photo_detail(self.conn, photo_id)

        self.assertEqual(detail["location"]["name"], "우리집")

    def test_no_location_is_none_not_an_error(self):
        photo_id = self._photo()
        self.assertIsNone(repository.photo_detail(self.conn, photo_id)["location"])


class ReassignFaceTests(PhotoEditTestCase):
    def test_moves_to_an_existing_named_cluster(self):
        photo_id = self._photo()
        face_id, original_cluster = self._face(photo_id, person=0)
        with self.conn:
            target_cluster = repository.insert_face_cluster(self.conn)
            repository.name_cluster(self.conn, "face", target_cluster, "아빠")

        with self.conn:
            result = repository.reassign_face(self.conn, face_id, "아빠")

        self.assertEqual(result, target_cluster)
        row = self.conn.execute("SELECT face_cluster_id, is_manual_override FROM face_instance WHERE id = ?", (face_id,)).fetchone()
        self.assertEqual(row["face_cluster_id"], target_cluster)
        self.assertEqual(row["is_manual_override"], 1)

    def test_creates_a_new_named_cluster_when_the_name_is_new(self):
        photo_id = self._photo()
        face_id, _ = self._face(photo_id)

        with self.conn:
            new_cluster_id = repository.reassign_face(self.conn, face_id, "새이름")

        row = repository.cluster_row(self.conn, "face", new_cluster_id)
        self.assertEqual(row["name"], "새이름")
        self.assertEqual(row["status"], "named")

    def test_empty_name_is_rejected(self):
        photo_id = self._photo()
        face_id, _ = self._face(photo_id)

        with self.assertRaises(ValueError):
            repository.reassign_face(self.conn, face_id, "   ")

    def test_reassign_route_redirects_back_to_the_photo(self):
        photo_id = self._photo()
        face_id, _ = self._face(photo_id)

        response = self.client.post(
            f"/photo/{photo_id}/face/{face_id}/reassign", data={"name": "아빠"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn(f"/photo/{photo_id}", response.headers["Location"])
        self.assertEqual(
            repository.photo_detail(self.conn, photo_id)["faces"][0]["cluster_name"], "아빠"
        )


class FalsePositiveTests(PhotoEditTestCase):
    def test_marking_excludes_from_the_cluster_but_keeps_the_row(self):
        photo_id = self._photo()
        face_id, cluster_id = self._face(photo_id)

        with self.conn:
            repository.mark_face_false_positive(self.conn, face_id)

        self.assertEqual(len(repository.face_instances_in_cluster(self.conn, cluster_id)), 0)
        # The row itself still exists — this is a flag, not a delete.
        row = self.conn.execute("SELECT id FROM face_instance WHERE id = ?", (face_id,)).fetchone()
        self.assertIsNotNone(row)

    def test_excluded_from_centroid_computation(self):
        # A false-positive detection (e.g. a misdetected pattern) must not
        # skew the cluster's centroid used for future assignment.
        photo_id = self._photo()
        cluster_id = None
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            real_face = repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0),
            )
            junk_face = repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1},
                embedding=_embedding(1),  # a very different "embedding" — the junk detection
            )
            repository.mark_face_false_positive(self.conn, junk_face)

        centroids = dict((cid, c) for cid, c, _n in repository.load_face_cluster_centroids(self.conn))
        # Centroid should equal the real face's embedding exactly, not a
        # blend with the excluded junk detection.
        self.assertEqual(centroids[cluster_id], _embedding(0))

    def test_restore_brings_it_back(self):
        photo_id = self._photo()
        face_id, cluster_id = self._face(photo_id)

        with self.conn:
            repository.mark_face_false_positive(self.conn, face_id)
            repository.restore_false_positive(self.conn, face_id)

        self.assertEqual(len(repository.face_instances_in_cluster(self.conn, cluster_id)), 1)

    def test_route_marks_and_redirects(self):
        photo_id = self._photo()
        face_id, cluster_id = self._face(photo_id)

        response = self.client.post(f"/photo/{photo_id}/face/{face_id}/false-positive")

        self.assertEqual(response.status_code, 302)
        self.assertEqual(repository.photo_detail(self.conn, photo_id)["faces"], [])


class AddManualFaceTests(PhotoEditTestCase):
    def test_creates_a_face_instance_marked_manual(self):
        photo_id = self._photo()

        with self.conn:
            cluster_id = repository.add_manual_face(
                self.conn, photo_id=photo_id, bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0), name="새사람",
            )

        row = repository.cluster_row(self.conn, "face", cluster_id)
        self.assertEqual(row["name"], "새사람")
        faces = repository.face_instances_in_cluster(self.conn, cluster_id)
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["photo_id"], photo_id)

    def test_assigns_to_an_existing_named_cluster_if_the_name_matches(self):
        photo_id_a = self._photo()
        photo_id_b = self._photo()
        _, existing_cluster = self._face(photo_id_a, cluster_name="언니")

        with self.conn:
            cluster_id = repository.add_manual_face(
                self.conn, photo_id=photo_id_b, bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0), name="언니",
            )

        self.assertEqual(cluster_id, existing_cluster)

    @patch("photo_grouping.face_embeddings.embed_manual_crop")
    def test_add_face_route_happy_path(self, mock_embed):
        photo_id = self._photo()
        mock_embed.return_value = (
            {"x": 0.12, "y": 0.12, "width": 0.2, "height": 0.2},
            _embedding(0),
        )

        response = self.client.post(
            f"/photo/{photo_id}/add-face",
            data={"x": "0.1", "y": "0.1", "width": "0.2", "height": "0.2", "name": "새사람"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(repository.photo_detail(self.conn, photo_id)["faces"]), 1)

    @patch("photo_grouping.face_embeddings.embed_manual_crop")
    def test_add_face_route_reports_when_no_face_found(self, mock_embed):
        photo_id = self._photo()
        mock_embed.return_value = None

        response = self.client.post(
            f"/photo/{photo_id}/add-face",
            data={"x": "0.1", "y": "0.1", "width": "0.05", "height": "0.05", "name": "새사람"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(repository.photo_detail(self.conn, photo_id)["faces"], [])

    def test_add_face_route_rejects_missing_name(self):
        photo_id = self._photo()

        response = self.client.post(
            f"/photo/{photo_id}/add-face",
            data={"x": "0.1", "y": "0.1", "width": "0.2", "height": "0.2", "name": ""},
        )

        self.assertEqual(response.status_code, 400)


class DateOverrideTests(PhotoEditTestCase):
    def test_sets_taken_at_override(self):
        photo_id = self._photo()

        with self.conn:
            repository.override_taken_at(self.conn, photo_id, "2020-01-01T12:00:00")

        row = self.conn.execute("SELECT taken_at, taken_at_override FROM photo WHERE id = ?", (photo_id,)).fetchone()
        self.assertEqual(row["taken_at_override"], "2020-01-01T12:00:00")
        self.assertEqual(row["taken_at"], "2026-04-12T10:00:00")  # original EXIF untouched

    def test_route_parses_datetime_local_format(self):
        photo_id = self._photo()

        response = self.client.post(
            f"/photo/{photo_id}/date", data={"taken_at": "2020-06-15T09:30"}
        )

        self.assertEqual(response.status_code, 302)
        row = self.conn.execute("SELECT taken_at_override FROM photo WHERE id = ?", (photo_id,)).fetchone()
        self.assertTrue(row["taken_at_override"].startswith("2020-06-15T09:30"))

    def test_route_rejects_garbage_input(self):
        photo_id = self._photo()

        response = self.client.post(f"/photo/{photo_id}/date", data={"taken_at": "not a date"})

        self.assertEqual(response.status_code, 400)


class LocationOverrideTests(PhotoEditTestCase):
    def test_creates_a_new_cluster_when_nothing_is_nearby(self):
        photo_id = self._photo()

        with self.conn:
            cluster_id = repository.override_photo_location(self.conn, photo_id, 37.5, 127.0)

        row = repository.cluster_row(self.conn, "place", cluster_id)
        self.assertAlmostEqual(row["centroid_lat"], 37.5)
        photo_location = self.conn.execute(
            "SELECT is_manual_override FROM photo_location WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        self.assertEqual(photo_location["is_manual_override"], 1)

    def test_joins_a_nearby_existing_cluster(self):
        photo_id = self._photo()
        with self.conn:
            existing = repository.insert_location_cluster(self.conn, lat=37.5000, lng=127.0000)

        with self.conn:
            cluster_id = repository.override_photo_location(self.conn, photo_id, 37.50001, 127.00001)

        self.assertEqual(cluster_id, existing)

    def test_overrides_an_existing_manual_assignment(self):
        # override_photo_location IS the manual correction, so unlike
        # set_photo_location() it must be allowed to replace a prior one.
        photo_id = self._photo()
        with self.conn:
            first = repository.override_photo_location(self.conn, photo_id, 10.0, 10.0)
            second = repository.override_photo_location(self.conn, photo_id, 50.0, 50.0)

        self.assertNotEqual(first, second)
        row = self.conn.execute(
            "SELECT location_cluster_id FROM photo_location WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        self.assertEqual(row["location_cluster_id"], second)

    def test_sets_the_gps_override_columns(self):
        photo_id = self._photo()

        with self.conn:
            repository.override_photo_location(self.conn, photo_id, 12.34, 56.78)

        row = self.conn.execute(
            "SELECT gps_override_lat, gps_override_lng FROM photo WHERE id = ?", (photo_id,)
        ).fetchone()
        self.assertAlmostEqual(row["gps_override_lat"], 12.34)
        self.assertAlmostEqual(row["gps_override_lng"], 56.78)

    def test_route_rejects_non_numeric_input(self):
        photo_id = self._photo()

        response = self.client.post(
            f"/photo/{photo_id}/location", data={"lat": "abc", "lng": "127.0"}
        )

        self.assertEqual(response.status_code, 400)

    def test_route_rejects_out_of_range_coordinates(self):
        photo_id = self._photo()

        response = self.client.post(
            f"/photo/{photo_id}/location", data={"lat": "999", "lng": "127.0"}
        )

        self.assertEqual(response.status_code, 400)

    def test_route_happy_path(self):
        photo_id = self._photo()

        response = self.client.post(
            f"/photo/{photo_id}/location", data={"lat": "37.5", "lng": "127.0"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertIsNotNone(repository.photo_detail(self.conn, photo_id)["location"])


class LocationByNameTests(PhotoEditTestCase):
    """§4.4: typing a place name instead of already knowing its
    coordinates — same underlying override_photo_location() path as
    LocationOverrideTests above, just reached via a name lookup first."""

    def test_route_rejects_blank_name(self):
        photo_id = self._photo()

        response = self.client.post(f"/photo/{photo_id}/location-by-name", data={"place_name": "  "})

        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.geocoding.forward_geocode")
    def test_route_saves_the_looked_up_coordinates(self, mock_forward):
        mock_forward.return_value = (37.5665, 126.978)
        photo_id = self._photo()

        response = self.client.post(
            f"/photo/{photo_id}/location-by-name", data={"place_name": "코롬방제과점"}
        )

        self.assertEqual(response.status_code, 302)
        mock_forward.assert_called_once_with("코롬방제과점")
        location = repository.photo_detail(self.conn, photo_id)["location"]
        self.assertIsNotNone(location)
        self.assertAlmostEqual(location["centroid_lat"], 37.5665)
        self.assertAlmostEqual(location["centroid_lng"], 126.978)

    @patch("photo_grouping.web.geocoding.forward_geocode")
    def test_a_geocode_miss_labels_the_photo_by_name_instead_of_erroring(self, mock_forward):
        # A real place with no map presence (a private venue, "우리집",
        # anywhere the geocoder just doesn't know) used to dead-end here
        # with a 400 telling the user to type coordinates they don't
        # have — now it's labeled by name only instead of failing.
        mock_forward.return_value = None
        photo_id = self._photo()

        response = self.client.post(
            f"/photo/{photo_id}/location-by-name", data={"place_name": "나만 아는 우리집"}
        )

        self.assertEqual(response.status_code, 302)
        location = repository.photo_detail(self.conn, photo_id)["location"]
        self.assertIsNotNone(location)
        self.assertEqual(location["name"], "나만 아는 우리집")
        self.assertEqual(location["centroid_lat"], repository.NO_COORDINATES)

    @patch("photo_grouping.web.geocoding.forward_geocode")
    def test_uses_the_same_override_path_as_coordinates(self, mock_forward):
        # Saving by name should behave identically to the coordinates
        # form — same manual-override flag, same gps_override_* columns —
        # not a parallel, subtly-different code path.
        mock_forward.return_value = (12.34, 56.78)
        photo_id = self._photo()

        self.client.post(f"/photo/{photo_id}/location-by-name", data={"place_name": "somewhere"})

        row = self.conn.execute(
            "SELECT gps_override_lat, gps_override_lng FROM photo WHERE id = ?", (photo_id,)
        ).fetchone()
        self.assertAlmostEqual(row["gps_override_lat"], 12.34)
        self.assertAlmostEqual(row["gps_override_lng"], 56.78)
        photo_location = self.conn.execute(
            "SELECT is_manual_override FROM photo_location WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        self.assertEqual(photo_location["is_manual_override"], 1)


class LocationWithoutCoordinatesTests(PhotoEditTestCase):
    """repository.set_photo_location_by_name_without_coordinates() and the
    NO_COORDINATES sentinel it uses — see that constant's docstring for
    why this is a sentinel rather than a nullable-column migration."""

    def test_labels_the_photo_with_the_given_name(self):
        photo_id = self._photo()

        with self.conn:
            repository.set_photo_location_by_name_without_coordinates(
                self.conn, photo_id, "나만 아는 우리집"
            )

        location = repository.photo_detail(self.conn, photo_id)["location"]
        self.assertEqual(location["name"], "나만 아는 우리집")
        self.assertEqual(location["centroid_lat"], repository.NO_COORDINATES)
        self.assertEqual(location["centroid_lng"], repository.NO_COORDINATES)
        self.assertEqual(location["is_manual_override"], 1)

    def test_reuses_an_existing_coordinate_less_cluster_with_the_same_name(self):
        photo_a = self._photo()
        photo_b = self._photo()

        with self.conn:
            id_a = repository.set_photo_location_by_name_without_coordinates(
                self.conn, photo_a, "나만 아는 우리집"
            )
            id_b = repository.set_photo_location_by_name_without_coordinates(
                self.conn, photo_b, "나만 아는 우리집"
            )

        self.assertEqual(id_a, id_b)

    def test_rejects_a_blank_name(self):
        photo_id = self._photo()

        with self.assertRaises(ValueError):
            with self.conn:
                repository.set_photo_location_by_name_without_coordinates(self.conn, photo_id, "   ")

    def test_excluded_from_the_spatial_clustering_candidate_list(self):
        # The whole point: a coordinate-less cluster must never be offered
        # to assign_or_create() as a nearest-centroid match for a real GPS
        # point — there's no real coordinate to compute a distance from.
        photo_id = self._photo()
        with self.conn:
            repository.set_photo_location_by_name_without_coordinates(
                self.conn, photo_id, "나만 아는 우리집"
            )

        clusters = repository.load_location_clusters(self.conn)

        self.assertEqual(clusters, [])

    def test_a_real_gps_photo_still_clusters_normally_alongside_a_coordinate_less_one(self):
        named_photo = self._photo()
        with self.conn:
            repository.set_photo_location_by_name_without_coordinates(
                self.conn, named_photo, "나만 아는 우리집"
            )

        gps_photo = self._photo()
        with self.conn:
            repository.override_photo_location(self.conn, gps_photo, 37.5665, 126.9780)

        location = repository.photo_detail(self.conn, gps_photo)["location"]
        self.assertAlmostEqual(location["centroid_lat"], 37.5665)
        self.assertAlmostEqual(location["centroid_lng"], 126.9780)

    def test_cluster_detail_page_hides_the_view_on_map_link(self):
        photo_id = self._photo()
        with self.conn:
            cluster_id = repository.set_photo_location_by_name_without_coordinates(
                self.conn, photo_id, "나만 아는 우리집"
            )

        response = self.client.get(f"/cluster/place/{cluster_id}")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"openstreetmap.org", response.data)

    def test_photo_detail_page_leaves_the_coordinate_fields_blank(self):
        photo_id = self._photo()
        with self.conn:
            repository.set_photo_location_by_name_without_coordinates(
                self.conn, photo_id, "나만 아는 우리집"
            )

        response = self.client.get(f"/photo/{photo_id}")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"999.000000", response.data)


class PageRenderTests(PhotoEditTestCase):
    def test_photo_detail_page_renders(self):
        photo_id = self._photo()
        self._face(photo_id, cluster_name="엄마")

        response = self.client.get(f"/photo/{photo_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("엄마".encode(), response.data)
        self.assertIn(b"face-box", response.data)

    def test_unknown_photo_is_404(self):
        self.assertEqual(self.client.get("/photo/9999").status_code, 404)

    def test_page_renders_the_location_map(self):
        photo_id = self._photo()

        response = self.client.get(f"/photo/{photo_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="location-map"', response.data)
        self.assertIn(b"leaflet.js", response.data)
        self.assertIn(b"leaflet.css", response.data)
        self.assertIn(b"initLocationMap", response.data)

    def test_location_map_static_files_are_served(self):
        for path in ("/static/leaflet/leaflet.js", "/static/leaflet/leaflet.css"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_edit_link_present_on_cluster_detail_photo_grid(self):
        photo_id = self._photo()
        face_id, cluster_id = self._face(photo_id)

        response = self.client.get(f"/cluster/face/{cluster_id}")

        self.assertIn(f"/photo/{photo_id}".encode(), response.data)


if __name__ == "__main__":
    unittest.main()
