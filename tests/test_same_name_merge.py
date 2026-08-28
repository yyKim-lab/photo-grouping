"""Same-name merge prompting.

A name the user has applied to two clusters is the strongest merge signal
available — they have *stated* the clusters are one person, where embedding
similarity only guesses. These cover surfacing that and acting on it.
"""

import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, repository, web  # noqa: E402

DIMS = 512


def _embedding(person: int) -> list[float]:
    vector = [0.0] * DIMS
    vector[person % DIMS] = 1.0
    return vector


class SameNameTestCase(unittest.TestCase):
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
        Image.new("RGB", (300, 200), color=(90, 90, 90)).save(path, "JPEG")
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


class DetectionTests(SameNameTestCase):
    def test_finds_a_name_used_on_two_clusters(self):
        self._face_cluster(3, person=0, name="Me")
        self._face_cluster(2, person=1, name="Me")
        self._face_cluster(1, person=2, name="엄마")

        duplicates = repository.duplicate_named_clusters(self.conn)

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["name"], "Me")
        self.assertEqual(duplicates[0]["instance_count"], 5)

    def test_unnamed_clusters_are_not_duplicates(self):
        self._face_cluster(2, person=0)
        self._face_cluster(2, person=1)

        self.assertEqual(repository.duplicate_named_clusters(self.conn), [])

    def test_clusters_sharing_a_name_orders_largest_first(self):
        small = self._face_cluster(1, person=0, name="Me")
        big = self._face_cluster(4, person=1, name="Me")

        shared = repository.clusters_sharing_a_name(self.conn, "face", "Me")

        self.assertEqual([c["id"] for c in shared], [big, small])


class PromptTests(SameNameTestCase):
    def test_naming_a_second_cluster_the_same_redirects_to_the_prompt(self):
        self._face_cluster(2, person=0, name="Me")
        second = self._face_cluster(2, person=1)

        response = self.client.post(f"/cluster/face/{second}/name", data={"name": "Me"})

        self.assertEqual(response.status_code, 302)
        self.assertIn("/same-name/face/", response.headers["Location"])

    def test_naming_a_unique_name_goes_straight_back_to_the_queue(self):
        cluster_id = self._face_cluster(2, person=0)

        response = self.client.post(f"/cluster/face/{cluster_id}/name", data={"name": "아빠"})

        self.assertIn("/queue", response.headers["Location"])

    def test_prompt_page_shows_every_cluster_with_that_name(self):
        self._face_cluster(2, person=0, name="Me")
        self._face_cluster(1, person=1, name="Me")

        response = self.client.get("/same-name/face/" + urllib.parse.quote("Me"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("그룹이 2개".encode(), response.data)  # ui_language defaults to Korean
        self.assertEqual(response.data.count(b"/crop"), 2)

    def test_prompt_redirects_away_when_only_one_cluster_has_the_name(self):
        self._face_cluster(2, person=0, name="Me")

        response = self.client.get("/same-name/face/Me")

        self.assertEqual(response.status_code, 302)


class MergeActionTests(SameNameTestCase):
    def test_merging_by_name_combines_every_matching_cluster(self):
        a = self._face_cluster(3, person=0, name="Me")
        self._face_cluster(2, person=1, name="Me")
        self._face_cluster(1, person=2, name="Me")

        response = self.client.post("/same-name/face/" + urllib.parse.quote("Me") + "/merge")

        self.assertEqual(response.status_code, 302)
        remaining = repository.clusters_sharing_a_name(self.conn, "face", "Me")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], a)  # largest survives
        self.assertEqual(remaining[0]["instance_count"], 6)

    def test_merge_returns_to_where_it_was_started_not_into_the_album(self):
        # Working through a list of duplicates shouldn't be interrupted by a
        # jump into one album; the album is offered as a link instead.
        self._face_cluster(3, person=0, name="Me")
        self._face_cluster(2, person=1, name="Me")

        response = self.client.post(
            "/same-name/face/" + urllib.parse.quote("Me") + "/merge",
            data={"next": "/review-duplicates"},
        )

        location = response.headers["Location"]
        self.assertTrue(location.startswith("/review-duplicates"))
        self.assertNotIn("/cluster/", location)

    def test_merge_passes_the_album_through_for_an_optional_link(self):
        a = self._face_cluster(3, person=0, name="Me")
        self._face_cluster(2, person=1, name="Me")

        response = self.client.post(
            "/same-name/face/" + urllib.parse.quote("Me") + "/merge",
            data={"next": "/review-duplicates"},
        )

        self.assertIn(f"merged_id={a}", response.headers["Location"])

    def test_the_notice_offers_the_album_without_forcing_it(self):
        a = self._face_cluster(3, person=0, name="Me")
        self._face_cluster(2, person=1, name="Me")
        self.client.post(
            "/same-name/face/" + urllib.parse.quote("Me") + "/merge",
            data={"next": "/review-duplicates"},
        )

        response = self.client.get(
            f"/review-duplicates?merged_kind=face&merged_id={a}&merged_name=Me"
        )

        # "앨범 보기" ("View album") — ui_language defaults to Korean, see
        # i18n.py/repository.LANGUAGES.
        self.assertIn("앨범 보기".encode(), response.data)
        self.assertIn(f'/cluster/face/{a}'.encode(), response.data)

    def test_no_notice_without_a_merge(self):
        self._face_cluster(2, person=0, name="Me")
        self._face_cluster(1, person=1, name="Me")

        response = self.client.get("/review-duplicates")

        self.assertNotIn("앨범 보기".encode(), response.data)

    def test_review_page_defaults_next_back_to_itself(self):
        self._face_cluster(2, person=0, name="Me")
        self._face_cluster(1, person=1, name="Me")

        response = self.client.get("/review-duplicates")

        self.assertIn(b'name="next" value="/review-duplicates"', response.data)

    def test_keeping_them_separate_changes_nothing(self):
        # "Keep separate" is just a link back to the queue — no state is
        # written, so the pairing stays visible on the albums page for
        # later. Genuinely different people sharing a name get renamed
        # instead, which resolves it naturally.
        self._face_cluster(2, person=0, name="수희")
        self._face_cluster(1, person=1, name="수희")

        self.client.get("/queue")

        self.assertEqual(len(repository.duplicate_named_clusters(self.conn)), 1)

    def test_albums_page_flags_the_duplicate(self):
        self._face_cluster(2, person=0, name="Me")
        self._face_cluster(1, person=1, name="Me")

        response = self.client.get("/")

        self.assertIn("두 개 이상의 그룹에 나뉘어 있을 수 있습니다".encode(), response.data)  # ui_language defaults to Korean
        self.assertIn("그룹 2개".encode(), response.data)

    def test_review_page_lists_duplicates(self):
        self._face_cluster(2, person=0, name="Me")
        self._face_cluster(1, person=1, name="Me")

        response = self.client.get("/review-duplicates")

        self.assertEqual(response.status_code, 200)
        # Quotes render as HTML entities (&#34;) via Jinja's auto-escaping,
        # so match around them rather than the literal quote character.
        self.assertIn("Me".encode(), response.data)
        self.assertIn("하나로 병합".encode(), response.data)  # ui_language defaults to Korean

    def test_review_page_is_empty_when_nothing_is_duplicated(self):
        self._face_cluster(2, person=0, name="Me")

        response = self.client.get("/review-duplicates")

        self.assertIn("병합할 항목이 없습니다".encode(), response.data)  # ui_language defaults to Korean


if __name__ == "__main__":
    unittest.main()
