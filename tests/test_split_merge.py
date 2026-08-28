"""Split and merge (§4.3) — repository logic and the routes that drive it."""

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


def _make_photo_file(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (400, 300), color=(100, 100, 100)).save(path, "JPEG")


class SplitMergeTestCase(unittest.TestCase):
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
        self._n += 1
        name = f"p{self._n}.jpg"
        path = self.tmp / name
        _make_photo_file(path)
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

    def _place_cluster(self, n_photos=1, lat=37.5, lng=127.0, name=None) -> int:
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=lat, lng=lng)
            for _ in range(n_photos):
                photo_id = self._photo()
                repository.set_photo_gps(self.conn, photo_id, lat, lng)
                repository.set_photo_location(
                    self.conn, photo_id=photo_id, location_cluster_id=cluster_id
                )
            if name:
                repository.name_cluster(self.conn, "place", cluster_id, name)
        return cluster_id

    def _faces_in(self, cluster_id) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) c FROM face_instance WHERE face_cluster_id = ?", (cluster_id,)
        ).fetchone()["c"]


class MergeFaceTests(SplitMergeTestCase):
    def test_absorbs_faces_and_removes_the_other_cluster(self):
        big = self._face_cluster(n_faces=3)
        small = self._face_cluster(n_faces=1)

        with self.conn:
            survivor = repository.merge_face_clusters(self.conn, [big, small])

        self.assertEqual(survivor, big)
        self.assertEqual(self._faces_in(big), 4)
        self.assertIsNone(repository.cluster_row(self.conn, "face", small))

    def test_named_cluster_survives_even_when_smaller(self):
        # §4.3 prefers the named one — keeping the name the user gave.
        big_unnamed = self._face_cluster(n_faces=5)
        small_named = self._face_cluster(n_faces=1, name="엄마")

        with self.conn:
            survivor = repository.merge_face_clusters(self.conn, [big_unnamed, small_named])

        self.assertEqual(survivor, small_named)
        self.assertEqual(repository.cluster_row(self.conn, "face", survivor)["name"], "엄마")
        self.assertEqual(self._faces_in(survivor), 6)

    def test_moved_faces_are_marked_manual_override(self):
        # Otherwise a later re-clustering pass would quietly undo the merge.
        keep = self._face_cluster(n_faces=2)
        absorb = self._face_cluster(n_faces=1)

        with self.conn:
            repository.merge_face_clusters(self.conn, [keep, absorb])

        overrides = self.conn.execute(
            "SELECT COUNT(*) c FROM face_instance WHERE is_manual_override = 1"
        ).fetchone()["c"]
        self.assertEqual(overrides, 1)

    def test_merging_three_clusters_at_once(self):
        a, b, c = self._face_cluster(3), self._face_cluster(1), self._face_cluster(1)

        with self.conn:
            survivor = repository.merge_face_clusters(self.conn, [a, b, c])

        self.assertEqual(survivor, a)
        self.assertEqual(self._faces_in(a), 5)

    def test_logs_a_cluster_event(self):
        a, b = self._face_cluster(2), self._face_cluster(1)

        with self.conn:
            survivor = repository.merge_face_clusters(self.conn, [a, b])

        event = repository.cluster_events(self.conn)[0]
        self.assertEqual(event["event_type"], "merge")
        self.assertEqual(event["cluster_type"], "face")
        self.assertEqual(event["source_cluster_ids"], sorted([a, b]))
        self.assertEqual(event["resulting_cluster_ids"], [survivor])

    def test_refuses_fewer_than_two(self):
        a = self._face_cluster(1)
        with self.assertRaises(ValueError):
            repository.merge_face_clusters(self.conn, [a])


class MergePlaceTests(SplitMergeTestCase):
    def test_recomputes_the_centroid_from_all_member_photos(self):
        # A stale centroid would misplace the merged cluster on the map and
        # skew future assignment.
        west = self._place_cluster(n_photos=1, lat=37.5, lng=127.0)
        east = self._place_cluster(n_photos=1, lat=37.5, lng=127.2)

        with self.conn:
            survivor = repository.merge_location_clusters(self.conn, [west, east])

        row = repository.cluster_row(self.conn, "place", survivor)
        self.assertAlmostEqual(row["centroid_lng"], 127.1, places=4)

    def test_logs_a_location_cluster_event(self):
        a, b = self._place_cluster(2), self._place_cluster(1, lat=37.6)

        with self.conn:
            repository.merge_location_clusters(self.conn, [a, b])

        event = repository.cluster_events(self.conn)[0]
        self.assertEqual(event["cluster_type"], "location")
        self.assertEqual(event["event_type"], "merge")


class SplitTests(SplitMergeTestCase):
    def test_moves_selected_faces_to_a_new_unlabeled_cluster(self):
        cluster_id = self._face_cluster(n_faces=4)
        face_ids = [
            r["id"]
            for r in self.conn.execute(
                "SELECT id FROM face_instance WHERE face_cluster_id = ? LIMIT 2", (cluster_id,)
            )
        ]

        with self.conn:
            new_id = repository.split_face_cluster(self.conn, cluster_id, face_ids)

        self.assertEqual(self._faces_in(cluster_id), 2)
        self.assertEqual(self._faces_in(new_id), 2)
        self.assertEqual(repository.cluster_row(self.conn, "face", new_id)["status"], "unlabeled")

    def test_refuses_to_split_out_every_face(self):
        # That would leave an empty husk and merely rename the cluster.
        cluster_id = self._face_cluster(n_faces=2)
        face_ids = [
            r["id"]
            for r in self.conn.execute(
                "SELECT id FROM face_instance WHERE face_cluster_id = ?", (cluster_id,)
            )
        ]

        with self.assertRaises(ValueError):
            repository.split_face_cluster(self.conn, cluster_id, face_ids)

    def test_logs_a_split_event_naming_both_resulting_clusters(self):
        cluster_id = self._face_cluster(n_faces=3)
        face_id = self.conn.execute(
            "SELECT id FROM face_instance WHERE face_cluster_id = ? LIMIT 1", (cluster_id,)
        ).fetchone()["id"]

        with self.conn:
            new_id = repository.split_face_cluster(self.conn, cluster_id, [face_id])

        event = repository.cluster_events(self.conn)[0]
        self.assertEqual(event["event_type"], "split")
        self.assertEqual(event["source_cluster_ids"], [cluster_id])
        self.assertEqual(sorted(event["resulting_cluster_ids"]), sorted([cluster_id, new_id]))


class CandidateRankingTests(SplitMergeTestCase):
    def test_ranks_the_same_person_first(self):
        target = self._face_cluster(n_faces=2, person=0)
        same_person = self._face_cluster(n_faces=1, person=0)  # identical embedding
        other = self._face_cluster(n_faces=1, person=7)  # orthogonal

        candidates = repository.similar_face_clusters(self.conn, target)

        self.assertEqual(candidates[0]["id"], same_person)
        self.assertAlmostEqual(candidates[0]["similarity"], 1.0, places=5)
        self.assertLess(
            next(c for c in candidates if c["id"] == other)["similarity"], 0.1
        )

    def test_excludes_itself(self):
        target = self._face_cluster(n_faces=1)
        self._face_cluster(n_faces=1, person=3)

        self.assertNotIn(target, [c["id"] for c in repository.similar_face_clusters(self.conn, target)])


class MergeRouteTests(SplitMergeTestCase):
    def test_merge_route_combines_and_redirects_to_survivor(self):
        big = self._face_cluster(n_faces=3)
        small = self._face_cluster(n_faces=1)

        response = self.client.post(
            f"/cluster/face/{small}/merge", data={"other_id": str(big)}
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn(f"/cluster/face/{big}", response.headers["Location"])
        self.assertEqual(self._faces_in(big), 4)

    def test_merge_with_nothing_selected_is_a_no_op(self):
        cluster_id = self._face_cluster(n_faces=2)

        self.client.post(f"/cluster/face/{cluster_id}/merge", data={})

        self.assertEqual(self._faces_in(cluster_id), 2)

    def test_split_route_redirects_to_the_new_cluster(self):
        cluster_id = self._face_cluster(n_faces=3)
        face_id = self.conn.execute(
            "SELECT id FROM face_instance WHERE face_cluster_id = ? LIMIT 1", (cluster_id,)
        ).fetchone()["id"]

        response = self.client.post(
            f"/cluster/face/{cluster_id}/split", data={"face_instance_id": str(face_id)}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(self._faces_in(cluster_id), 2)

    def test_splitting_everything_shows_an_error_not_a_crash(self):
        cluster_id = self._face_cluster(n_faces=2)
        face_ids = [
            str(r["id"])
            for r in self.conn.execute(
                "SELECT id FROM face_instance WHERE face_cluster_id = ?", (cluster_id,)
            )
        ]

        response = self.client.post(
            f"/cluster/face/{cluster_id}/split", data={"face_instance_id": face_ids}
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self._faces_in(cluster_id), 2)

    def test_detail_page_offers_merge_candidates(self):
        target = self._face_cluster(n_faces=1, person=0)
        self._face_cluster(n_faces=1, person=0)

        response = self.client.get(f"/cluster/face/{target}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("이 중에 같은 사람이 있나요?".encode(), response.data)  # ui_language defaults to Korean


if __name__ == "__main__":
    unittest.main()
