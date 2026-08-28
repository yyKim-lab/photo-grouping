"""Place-name suggestions: repository storage + UI display of geocoded and
OCR candidates (§3 step 6). Neither source ever writes LocationCluster.name
— these check that boundary holds, plus the display fallback from raw
coordinates to a suggested name.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402


class PlaceSuggestionTestCase(unittest.TestCase):
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
        Image.new("RGB", (200, 200), color=(60, 60, 60)).save(path, "JPEG")
        with self.conn:
            return repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at="2026-04-12T10:00:00",
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )

    def _place_cluster(self, n_photos=1, lat=34.78998, lng=126.38450) -> int:
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=lat, lng=lng)
            for _ in range(n_photos):
                repository.set_photo_location(
                    self.conn, photo_id=self._photo(), location_cluster_id=cluster_id
                )
        return cluster_id


class StorageTests(PlaceSuggestionTestCase):
    def test_stores_and_reads_back_a_geocoded_name(self):
        cluster_id = self._place_cluster()

        with self.conn:
            repository.set_location_geocoded_name(self.conn, cluster_id, "코롬방제과점, 목포시")

        row = repository.cluster_row(self.conn, "place", cluster_id)
        self.assertEqual(row["geocoded_name"], "코롬방제과점, 목포시")
        self.assertIsNotNone(row["geocoded_at"])

    def test_records_the_timestamp_even_when_geocoding_found_nothing(self):
        # So a cluster with no answer isn't retried on every run.
        cluster_id = self._place_cluster()

        with self.conn:
            repository.set_location_geocoded_name(self.conn, cluster_id, None)

        row = repository.cluster_row(self.conn, "place", cluster_id)
        self.assertIsNone(row["geocoded_name"])
        self.assertIsNotNone(row["geocoded_at"])

    def test_naming_the_cluster_does_not_touch_the_suggestion_columns(self):
        # The suggestion and the user's actual choice are independent —
        # geocoded_name must survive the user naming (or renaming) the place.
        cluster_id = self._place_cluster()
        with self.conn:
            repository.set_location_geocoded_name(self.conn, cluster_id, "코롬방제과점, 목포시")
            repository.name_cluster(self.conn, "place", cluster_id, "코롬방")

        row = repository.cluster_row(self.conn, "place", cluster_id)
        self.assertEqual(row["name"], "코롬방")
        self.assertEqual(row["geocoded_name"], "코롬방제과점, 목포시")

    def test_needing_suggestions_excludes_clusters_already_processed(self):
        done = self._place_cluster()
        pending = self._place_cluster()
        with self.conn:
            repository.set_location_geocoded_name(self.conn, done, "Somewhere")

        remaining = repository.location_clusters_needing_suggestions(self.conn, source="geocode")

        self.assertEqual([c["id"] for c in remaining], [pending])

    def test_geocode_and_ocr_progress_are_tracked_independently(self):
        # A cluster can have a geocoded name but no OCR pass yet, or vice
        # versa — they run as separate passes over separate needs-lists.
        cluster_id = self._place_cluster()
        with self.conn:
            repository.set_location_geocoded_name(self.conn, cluster_id, "Somewhere")

        self.assertEqual(repository.location_clusters_needing_suggestions(self.conn, source="geocode"), [])
        self.assertEqual(len(repository.location_clusters_needing_suggestions(self.conn, source="ocr")), 1)


class DisplayTests(PlaceSuggestionTestCase):
    def test_queue_card_shows_suggested_name_not_raw_coordinates(self):
        cluster_id = self._place_cluster()
        with self.conn:
            repository.set_location_geocoded_name(self.conn, cluster_id, "코롬방제과점, 목포시")

        response = self.client.get("/queue")

        self.assertIn("코롬방제과점".encode(), response.data)
        self.assertNotIn(b"34.78998, 126.38450", response.data)

    def test_queue_card_falls_back_to_unknown_without_any_suggestion(self):
        self._place_cluster()

        response = self.client.get("/queue")

        self.assertIn("위치 정보 없음".encode(), response.data)  # ui_language defaults to Korean

    def test_ocr_candidates_offered_as_chips(self):
        cluster_id = self._place_cluster()
        with self.conn:
            repository.set_location_ocr_name(
                self.conn, cluster_id, json.dumps([{"text": "코롬방", "confidence": 0.78}])
            )

        response = self.client.get("/queue")

        self.assertIn("코롬방".encode(), response.data)
        self.assertIn(b"sign", response.data)  # source label

    def test_map_view_link_is_still_present(self):
        # The suggestion is a name; the coordinates should stay reachable
        # via the map link rather than disappearing entirely.
        self._place_cluster()

        response = self.client.get("/queue")

        self.assertIn("지도에서 보기".encode(), response.data)  # ui_language defaults to Korean

    def test_cluster_detail_shows_suggestion_chips(self):
        cluster_id = self._place_cluster()
        with self.conn:
            repository.set_location_ocr_name(
                self.conn, cluster_id, json.dumps([{"text": "코롬방", "confidence": 0.78}])
            )

        response = self.client.get(f"/cluster/place/{cluster_id}")

        self.assertIn("코롬방".encode(), response.data)

    def test_ocr_suggestion_ranks_before_map_suggestion(self):
        # A storefront sign usually names a venue better than an address.
        cluster_id = self._place_cluster()
        with self.conn:
            repository.set_location_ocr_name(
                self.conn, cluster_id, json.dumps([{"text": "코롬방", "confidence": 0.78}])
            )
            repository.set_location_geocoded_name(self.conn, cluster_id, "영산로75번길, 목포시")

        row = repository.cluster_row(self.conn, "place", cluster_id)
        suggestions = web._name_suggestions(row)

        self.assertEqual(suggestions[0]["text"], "코롬방")
        self.assertEqual(suggestions[0]["source"], "sign")


if __name__ == "__main__":
    unittest.main()
