"""Ingestion pipeline tests — no network, no Pillow, no face_recognition.

The Picker client and face detector are injected, so these exercise the
real database writes and clustering decisions against fakes. EXIF parsing
is stubbed out for the same reason it lives in its own test file: it needs
Pillow, which isn't in the stdlib environment the rest of the suite runs in.
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db, face_clustering, ingestion, repository  # noqa: E402


class FakePickerClient:
    def __init__(self, fail_on: set[str] | None = None):
        self.fail_on = fail_on or set()
        self.original_fetches: list[str] = []
        self.processing_fetches: list[str] = []

    def fetch_processing_bytes(self, access_token, base_url):
        if base_url in self.fail_on:
            raise RuntimeError("simulated transient fetch failure")
        self.processing_fetches.append(base_url)
        return f"processing:{base_url}".encode()

    def fetch_original_bytes(self, access_token, base_url):
        if base_url in self.fail_on:
            raise RuntimeError("simulated transient fetch failure")
        self.original_fetches.append(base_url)
        return f"original:{base_url}".encode()


class FakeStorage:
    backend_name = "local"

    def __init__(self):
        self.saved: dict[str, bytes] = {}

    def save_original(self, data, filename):
        self.saved[filename] = data
        return self.backend_name, f"/fake/originals/{filename}"


def _item(media_id, filename, base_url=None):
    return {
        "id": media_id,
        "mediaFile": {"filename": filename, "baseUrl": base_url or f"https://x/{media_id}"},
    }


def _embedding(person: int, dims: int = 512) -> list[float]:
    """A unit vector standing in for one person's ArcFace embedding.

    Distinct `person` values are orthogonal (cosine similarity 0, far below
    the clustering threshold) while the same value is identical (similarity
    1), which is all these tests need. Note a constant vector like
    [0.0]*dims would be degenerate under cosine similarity — it has no
    direction — so each person gets a distinct basis direction instead."""
    vector = [0.0] * dims
    vector[person % dims] = 1.0
    return vector


class IngestionTestCase(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")
        db.migrate(self.conn)
        self.picker = FakePickerClient()
        self.storage = FakeStorage()
        # extract_taken_at needs Pillow; the pipeline's own fallback chain
        # is what's under test elsewhere, so pin a deterministic value here.
        self._exif_patch = patch(
            "photo_grouping.ingestion.exif.extract_taken_at",
            return_value=datetime(2026, 4, 12, 10, 30, 0),
        )
        self._exif_patch.start()
        # No GPS by default; local-import tests override this per-test —
        # picker-flow tests never touch it either way (GPS is always None
        # at ingestion time on that path, per the module docstring).
        self._gps_patch = patch("photo_grouping.ingestion.exif.extract_gps", return_value=None)
        self._gps_patch.start()

    def tearDown(self):
        self._exif_patch.stop()
        self._gps_patch.stop()
        self.conn.close()

    def _ingest(self, items, detect_faces=None, **kwargs):
        return ingestion.ingest_picked_items(
            self.conn,
            items,
            access_token="token",
            picker_client=self.picker,
            storage_adapter=self.storage,
            detect_faces=detect_faces,
            **kwargs,
        )

    def _ingest_local(self, files, detect_faces=None, **kwargs):
        return ingestion.ingest_local_files(
            self.conn,
            files,
            storage_adapter=self.storage,
            detect_faces=detect_faces,
            **kwargs,
        )


class IngestPickedItemsTests(IngestionTestCase):
    def test_imports_photo_rows_and_saves_originals(self):
        result = self._ingest([_item("m1", "a.jpg"), _item("m2", "b.jpg")])

        self.assertEqual(result.imported, 2)
        self.assertEqual(result.failed, [])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 2)
        self.assertIn("a.jpg", self.storage.saved)

        row = self.conn.execute("SELECT * FROM photo WHERE picker_media_id = 'm1'").fetchone()
        self.assertEqual(row["original_storage_backend"], "local")
        self.assertEqual(row["original_storage_path"], "/fake/originals/a.jpg")

    def test_gps_is_null_at_ingestion_time(self):
        # The Picker API can't supply GPS; it arrives later via Takeout.
        self._ingest([_item("m1", "a.jpg")])

        row = self.conn.execute("SELECT gps_lat, gps_lng FROM photo").fetchone()
        self.assertIsNone(row["gps_lat"])
        self.assertIsNone(row["gps_lng"])

    def test_rerunning_skips_already_imported_photos(self):
        self._ingest([_item("m1", "a.jpg")])
        result = self._ingest([_item("m1", "a.jpg"), _item("m2", "b.jpg")])

        self.assertEqual(result.imported, 1)
        self.assertEqual(result.skipped_already_imported, 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 2)

    def test_one_failure_does_not_abort_the_batch(self):
        self.picker.fail_on = {"https://x/m2"}

        result = self._ingest([_item("m1", "a.jpg"), _item("m2", "b.jpg"), _item("m3", "c.jpg")])

        self.assertEqual(result.imported, 2)
        self.assertEqual(len(result.failed), 1)
        self.assertEqual(result.failed[0][0], "b.jpg")
        self.assertEqual(result.attempted, 3)

    def test_failed_photo_leaves_no_partial_row(self):
        self.picker.fail_on = {"https://x/m1"}

        self._ingest([_item("m1", "a.jpg")])

        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 0)

    def test_a_failed_photo_can_be_retried_by_rerunning(self):
        self.picker.fail_on = {"https://x/m1"}
        self._ingest([_item("m1", "a.jpg")])

        self.picker.fail_on = set()
        result = self._ingest([_item("m1", "a.jpg")])

        self.assertEqual(result.imported, 1)


class IngestLocalFilesTests(IngestionTestCase):
    """Local-device import (no Google/OAuth involved at all) — the escape
    hatch from the whole Picker session/polling flow."""

    def test_imports_photo_rows_and_saves_originals(self):
        result = self._ingest_local([("a.jpg", b"fake-bytes-a"), ("b.jpg", b"fake-bytes-b")])

        self.assertEqual(result.imported, 2)
        self.assertEqual(result.failed, [])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 2)
        self.assertIn("a.jpg", self.storage.saved)

    def test_dedup_is_by_content_not_filename(self):
        self._ingest_local([("a.jpg", b"same-bytes")])
        result = self._ingest_local([("a-renamed.jpg", b"same-bytes")])

        self.assertEqual(result.imported, 0)
        self.assertEqual(result.skipped_already_imported, 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 1)

    def test_reuploading_the_identical_file_is_a_no_op(self):
        self._ingest_local([("a.jpg", b"same-bytes")])
        result = self._ingest_local([("a.jpg", b"same-bytes")])

        self.assertEqual(result.skipped_already_imported, 1)

    def test_different_files_with_the_same_name_both_import(self):
        # Content-based dedup means a filename collision alone isn't a dup.
        self._ingest_local([("a.jpg", b"bytes-one")])
        result = self._ingest_local([("a.jpg", b"bytes-two")])

        self.assertEqual(result.imported, 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo").fetchone()["c"], 2)

    def test_gps_is_read_directly_no_takeout_needed(self):
        with patch("photo_grouping.ingestion.exif.extract_gps", return_value=(37.5, 127.0)):
            self._ingest_local([("a.jpg", b"bytes")])

        row = self.conn.execute("SELECT gps_lat, gps_lng FROM photo").fetchone()
        self.assertEqual((row["gps_lat"], row["gps_lng"]), (37.5, 127.0))

    def test_gps_photos_are_clustered_inline_no_separate_pass_needed(self):
        with patch("photo_grouping.ingestion.exif.extract_gps", return_value=(37.5, 127.0)):
            self._ingest_local([("a.jpg", b"bytes")])

        row = self.conn.execute(
            "SELECT location_cluster_id FROM photo_location pl "
            "JOIN photo p ON p.id = pl.photo_id"
        ).fetchone()
        self.assertIsNotNone(row["location_cluster_id"])

    def test_photo_without_gps_is_not_clustered(self):
        self._ingest_local([("a.jpg", b"bytes")])  # extract_gps mocked to None in setUp

        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo_location").fetchone()["c"], 0)

    def test_one_failure_does_not_abort_the_batch(self):
        class FailingStorage(FakeStorage):
            def save_original(self, data, filename):
                if filename == "b.jpg":
                    raise RuntimeError("simulated disk failure")
                return super().save_original(data, filename)

        self.storage = FailingStorage()
        result = self._ingest_local([("a.jpg", b"1"), ("b.jpg", b"2"), ("c.jpg", b"3")])

        self.assertEqual(result.imported, 2)
        self.assertEqual(len(result.failed), 1)
        self.assertEqual(result.failed[0][0], "b.jpg")

    def test_detects_and_clusters_faces_same_as_the_picker_path(self):
        def detect_faces(image_bytes):
            return [({"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}, _embedding(0))]

        result = self._ingest_local([("a.jpg", b"bytes")], detect_faces=detect_faces)

        self.assertEqual(result.faces_detected, 1)
        self.assertEqual(result.face_clusters_created, 1)


class SeedMatchingDuringIngestionTests(IngestionTestCase):
    """§3 step 5: a seed-face match must only ever produce a suggestion on
    the new cluster, never an auto-assigned name — seed embeddings are
    explicitly less reliable than full-photo ones."""

    def test_matching_face_gets_a_suggested_name_not_an_assigned_one(self):
        with self.conn:
            repository.insert_seed_face(self.conn, name="엄마", embedding=_embedding(0))

        def detect(image_bytes):
            return [({"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}, _embedding(0))]

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)

        cluster = self.conn.execute("SELECT status, name, suggested_name FROM face_cluster").fetchone()
        self.assertEqual(cluster["status"], "unlabeled")
        self.assertIsNone(cluster["name"])
        self.assertEqual(cluster["suggested_name"], "엄마")

    def test_non_matching_face_gets_no_suggestion(self):
        with self.conn:
            repository.insert_seed_face(self.conn, name="엄마", embedding=_embedding(0))

        def detect(image_bytes):
            return [({"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}, _embedding(1))]

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)

        cluster = self.conn.execute("SELECT suggested_name FROM face_cluster").fetchone()
        self.assertIsNone(cluster["suggested_name"])

    def test_matching_an_existing_cluster_does_not_touch_seed_matching(self):
        # A face that joins an already-existing cluster shouldn't run the
        # seed check at all — seed matching only applies to a brand-new
        # cluster; an existing one already has (or doesn't have) a name.
        with self.conn:
            repository.insert_seed_face(self.conn, name="엄마", embedding=_embedding(0))

        def detect(image_bytes):
            return [({"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}, _embedding(0))]

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)
        self._ingest([_item("m2", "b.jpg")], detect_faces=detect)

        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM face_cluster").fetchone()["c"], 1)

    def test_no_seed_faces_means_no_query_overhead_when_not_detecting(self):
        # detect_faces=None (a face-less ingestion run) shouldn't touch
        # seed_face at all.
        result = self._ingest([_item("m1", "a.jpg")])
        self.assertEqual(result.faces_detected, 0)


class FaceClusteringDuringIngestionTests(IngestionTestCase):
    def test_similar_faces_join_one_cluster_distinct_faces_do_not(self):
        def detect(image_bytes):
            # Two different people in every photo — orthogonal
            # embeddings, so they must never merge into one cluster.
            return [
                ({"x": 0, "y": 0, "width": 10, "height": 10}, _embedding(0)),
                ({"x": 1, "y": 1, "width": 10, "height": 10}, _embedding(1)),
            ]

        result = self._ingest([_item("m1", "a.jpg"), _item("m2", "b.jpg")], detect_faces=detect)

        self.assertEqual(result.faces_detected, 4)
        # Two people across two photos -> two clusters, not four.
        self.assertEqual(result.face_clusters_created, 2)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM face_cluster").fetchone()["c"], 2)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM face_instance").fetchone()["c"], 4)

    def test_faces_are_detected_on_the_original_not_a_second_fetch(self):
        # The separate processing fetch was removed: the original is
        # downloaded for storage regardless, so fetching a downscaled copy
        # of the same photo doubled API calls for nothing — and detecting
        # at a different resolution than the distance threshold was tuned
        # at silently changes clustering.
        seen = []

        def detect(image_bytes):
            seen.append(image_bytes)
            return []

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)

        self.assertEqual(seen, [b"original:https://x/m1"])
        self.assertEqual(self.picker.processing_fetches, [])
        self.assertEqual(self.picker.original_fetches, ["https://x/m1"])

    def test_new_clusters_start_unlabeled_with_a_representative_photo(self):
        def detect(image_bytes):
            return [({"x": 0, "y": 0, "width": 1, "height": 1}, _embedding(0))]

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)

        cluster = self.conn.execute("SELECT * FROM face_cluster").fetchone()
        self.assertEqual(cluster["status"], "unlabeled")
        self.assertIsNotNone(cluster["representative_photo_id"])

    def test_embedding_survives_the_storage_roundtrip(self):
        original = [i / 1000.0 for i in range(128)]

        def detect(image_bytes):
            return [({"x": 0, "y": 0, "width": 1, "height": 1}, original)]

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)

        blob = self.conn.execute("SELECT embedding FROM face_instance").fetchone()["embedding"]
        decoded = repository.decode_embedding(blob)
        self.assertEqual(len(decoded), 128)
        for expected, actual in zip(original, decoded):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_clusters_persist_across_separate_ingestion_runs(self):
        def detect(image_bytes):
            return [({"x": 0, "y": 0, "width": 1, "height": 1}, _embedding(0))]

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)
        result = self._ingest([_item("m2", "b.jpg")], detect_faces=detect)

        # The second run must recognise the first run's cluster, not make
        # a new one — this is what load_face_cluster_centroids() is for.
        self.assertEqual(result.face_clusters_created, 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM face_cluster").fetchone()["c"], 1)

    def test_bounding_box_is_stored_as_json(self):
        box = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}

        def detect(image_bytes):
            return [(box, _embedding(0))]

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)

        stored = self.conn.execute("SELECT bounding_box FROM face_instance").fetchone()["bounding_box"]
        self.assertEqual(json.loads(stored), box)


class LocationClusteringTests(IngestionTestCase):
    def _photo_with_gps(self, media_id, filename, lat, lng):
        self._ingest([_item(media_id, filename)])
        photo_id = repository.photo_id_for_picker_media_id(self.conn, media_id)
        with self.conn:
            repository.set_photo_gps(self.conn, photo_id, lat, lng)
        return photo_id

    def test_nearby_photos_share_a_cluster_distant_ones_do_not(self):
        self._photo_with_gps("m1", "a.jpg", 37.5000, 127.0000)
        self._photo_with_gps("m2", "b.jpg", 37.50045, 127.0000)  # ~50m away
        self._photo_with_gps("m3", "c.jpg", 35.1796, 129.0756)  # Busan

        created = ingestion.cluster_locations(self.conn)

        self.assertEqual(created, 2)
        rows = self.conn.execute(
            "SELECT location_cluster_id, COUNT(*) c FROM photo_location GROUP BY location_cluster_id"
        ).fetchall()
        self.assertEqual(sorted(r["c"] for r in rows), [1, 2])

    def test_photos_without_gps_are_left_unassigned(self):
        self._ingest([_item("m1", "no_gps.jpg")])

        ingestion.cluster_locations(self.conn)

        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM photo_location").fetchone()["c"], 0)

    def test_manual_location_override_is_not_clobbered(self):
        photo_id = self._photo_with_gps("m1", "a.jpg", 37.5, 127.0)
        with self.conn:
            other_cluster = repository.insert_location_cluster(self.conn, lat=1.0, lng=1.0)
            self.conn.execute(
                """
                INSERT INTO photo_location (photo_id, location_cluster_id, is_manual_override)
                VALUES (?, ?, 1)
                """,
                (photo_id, other_cluster),
            )

        ingestion.cluster_locations(self.conn)

        assigned = self.conn.execute(
            "SELECT location_cluster_id FROM photo_location WHERE photo_id = ?", (photo_id,)
        ).fetchone()["location_cluster_id"]
        self.assertEqual(assigned, other_cluster)

    def test_rerunning_does_not_duplicate_clusters(self):
        self._photo_with_gps("m1", "a.jpg", 37.5, 127.0)

        ingestion.cluster_locations(self.conn)
        created_again = ingestion.cluster_locations(self.conn)

        self.assertEqual(created_again, 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) c FROM location_cluster").fetchone()["c"], 1)


class TakeoutBackfillTests(IngestionTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.takeout_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def _write_takeout_photo(self, filename, lat, lng):
        (self.takeout_dir / filename).write_bytes(b"fake")
        (self.takeout_dir / f"{filename}.json").write_text(
            json.dumps({"geoData": {"latitude": lat, "longitude": lng, "altitude": 0.0}})
        )

    def test_backfills_gps_by_filename(self):
        self._ingest([_item("m1", "a.jpg")])
        self._write_takeout_photo("a.jpg", 37.5, 127.0)

        result = ingestion.backfill_gps_from_takeout(self.conn, self.takeout_dir)

        self.assertEqual(result.matched, 1)
        row = self.conn.execute("SELECT gps_lat, gps_lng FROM photo").fetchone()
        self.assertAlmostEqual(row["gps_lat"], 37.5)
        self.assertAlmostEqual(row["gps_lng"], 127.0)

    def test_matches_on_original_filename_not_dedup_suffixed_path(self):
        # Regression test. Storage adapters append "_1" when two photos
        # arrive with the same name (common: IMG_0001.jpg from two
        # devices). Deriving the Takeout lookup key from the stored path
        # therefore searched for "a_1.jpg", which no sidecar ever matches,
        # and GPS backfill silently found nothing.
        class CollidingStorage(FakeStorage):
            def save_original(self, data, filename):
                stem, _, ext = filename.rpartition(".")
                return self.backend_name, f"/fake/originals/{stem}_1.{ext}"

        self.storage = CollidingStorage()
        self._ingest([_item("m1", "a.jpg")])
        self._write_takeout_photo("a.jpg", 37.5, 127.0)

        result = ingestion.backfill_gps_from_takeout(self.conn, self.takeout_dir)

        self.assertEqual(result.matched, 1)
        self.assertEqual(result.unmatched, [])

    def test_photo_with_no_takeout_record_is_reported_not_errored(self):
        self._ingest([_item("m1", "missing.jpg")])

        result = ingestion.backfill_gps_from_takeout(self.conn, self.takeout_dir)

        self.assertEqual(result.matched, 0)
        self.assertEqual(result.unmatched, ["missing.jpg"])

    def test_rerunning_skips_photos_that_already_have_gps(self):
        self._ingest([_item("m1", "a.jpg")])
        self._write_takeout_photo("a.jpg", 37.5, 127.0)

        ingestion.backfill_gps_from_takeout(self.conn, self.takeout_dir)
        second = ingestion.backfill_gps_from_takeout(self.conn, self.takeout_dir)

        self.assertEqual(second.matched, 0)

    def test_manual_gps_override_is_not_overwritten(self):
        self._ingest([_item("m1", "a.jpg")])
        photo_id = repository.photo_id_for_picker_media_id(self.conn, "m1")
        with self.conn:
            self.conn.execute(
                "UPDATE photo SET gps_override_lat = 1.0, gps_override_lng = 2.0 WHERE id = ?",
                (photo_id,),
            )
        self._write_takeout_photo("a.jpg", 37.5, 127.0)

        ingestion.backfill_gps_from_takeout(self.conn, self.takeout_dir)

        row = self.conn.execute("SELECT gps_override_lat, gps_override_lng FROM photo").fetchone()
        self.assertEqual(row["gps_override_lat"], 1.0)
        self.assertEqual(row["gps_override_lng"], 2.0)


class AnomalyInputTests(IngestionTestCase):
    def test_override_columns_take_precedence(self):
        self._ingest([_item("m1", "a.jpg")])
        photo_id = repository.photo_id_for_picker_media_id(self.conn, "m1")
        with self.conn:
            repository.set_photo_gps(self.conn, photo_id, 37.5, 127.0)
            self.conn.execute(
                """
                UPDATE photo
                SET gps_override_lat = 10.0, gps_override_lng = 20.0,
                    taken_at_override = '2020-01-01T00:00:00'
                WHERE id = ?
                """,
                (photo_id,),
            )

        photos = repository.load_photos_for_anomaly_check(self.conn)

        self.assertEqual(len(photos), 1)
        self.assertEqual(photos[0]["lat"], 10.0)
        self.assertEqual(photos[0]["lng"], 20.0)
        self.assertEqual(photos[0]["taken_at"], "2020-01-01T00:00:00")

    def test_includes_face_clusters_per_photo(self):
        def detect(image_bytes):
            return [({"x": 0, "y": 0, "width": 1, "height": 1}, _embedding(0))]

        self._ingest([_item("m1", "a.jpg")], detect_faces=detect)
        photo_id = repository.photo_id_for_picker_media_id(self.conn, "m1")
        with self.conn:
            repository.set_photo_gps(self.conn, photo_id, 37.5, 127.0)

        photos = repository.load_photos_for_anomaly_check(self.conn)

        self.assertEqual(len(photos[0]["face_cluster_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
