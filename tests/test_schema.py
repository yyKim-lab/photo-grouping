"""Sanity checks for the initial schema migration.

Not a full test suite for the app (there's no app logic yet) — just enough
to confirm the schema itself is sound: migrations apply cleanly and are
idempotent, foreign keys are enforced, CHECK constraints reject bad enum
values, and the PhotoLocation one-row-per-photo shape actually holds.

Run with: python -m unittest tests/test_schema.py -v
(stdlib unittest — no third-party test runner needed at this stage.)
"""

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import db  # noqa: E402


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_migrate_applies_and_is_idempotent(self):
        applied_first = db.migrate(self.conn)
        self.assertEqual(
            applied_first,
            [
                "0001_initial_schema.sql",
                "0002_photo_original_filename.sql",
                "0003_location_cluster_last_shown.sql",
                "0004_cluster_excluded_at.sql",
                "0005_location_name_suggestions.sql",
                "0006_face_instance_false_positive.sql",
                "0007_events_and_descriptions.sql",
                "0008_face_cluster_suggested_name.sql",
                "0009_pending_import_session.sql",
                "0010_autobio_settings.sql",
                "0011_app_settings.sql",
                "0012_narrative_and_ui_language.sql",
                "0013_llm_provider.sql",
                "0014_originals_dir.sql",
                "0015_event_autobio_exclude.sql",
            ],
        )

        applied_second = db.migrate(self.conn)
        self.assertEqual(applied_second, [])

    def test_migration_0002_backfills_original_filename_from_path(self):
        # Apply only the initial schema, insert a pre-0002 row, then let
        # 0002 run — the backfill should recover the basename.
        initial = (db.MIGRATIONS_DIR / "0001_initial_schema.sql").read_text()
        self.conn.executescript(initial)
        self.conn.execute(
            """
            INSERT INTO photo (picker_media_id, taken_at, original_storage_backend, original_storage_path)
            VALUES ('legacy', '2026-01-01T00:00:00Z', 'local', '/photos/nested/dir/legacy.jpg')
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        self.conn.execute("INSERT INTO schema_migrations (version) VALUES ('0001_initial_schema.sql')")
        self.conn.commit()

        db.migrate(self.conn)

        row = self.conn.execute("SELECT original_filename FROM photo WHERE picker_media_id = 'legacy'").fetchone()
        self.assertEqual(row["original_filename"], "legacy.jpg")

    def test_all_expected_tables_exist(self):
        db.migrate(self.conn)
        tables = {
            row["name"]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        expected = {
            "photo",
            "face_cluster",
            "face_instance",
            "location_cluster",
            "photo_location",
            "cluster_event",
            "seed_face",
            "autobio_entry",
            "autobio_summary",
            "schema_migrations",
        }
        self.assertTrue(expected.issubset(tables), tables)

    def _insert_photo(self, picker_media_id="pmi-1"):
        cur = self.conn.execute(
            """
            INSERT INTO photo (picker_media_id, taken_at, original_storage_backend, original_storage_path)
            VALUES (?, '2026-08-24T10:00:00Z', 'local', '/photos/a.jpg')
            """,
            (picker_media_id,),
        )
        return cur.lastrowid

    def test_photo_storage_backend_check_constraint(self):
        db.migrate(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                """
                INSERT INTO photo (picker_media_id, taken_at, original_storage_backend, original_storage_path)
                VALUES ('pmi-x', '2026-08-24T10:00:00Z', 'not_a_backend', '/x')
                """
            )

    def test_face_cluster_status_check_constraint(self):
        db.migrate(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO face_cluster (status) VALUES ('bogus')"
            )

    def test_face_instance_requires_valid_photo_and_cluster(self):
        db.migrate(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                """
                INSERT INTO face_instance (photo_id, face_cluster_id, bounding_box, embedding)
                VALUES (9999, 9999, '{}', X'00')
                """
            )

    def test_face_instance_cascades_on_photo_delete(self):
        db.migrate(self.conn)
        photo_id = self._insert_photo()
        cluster_id = self.conn.execute(
            "INSERT INTO face_cluster DEFAULT VALUES"
        ).lastrowid
        self.conn.execute(
            """
            INSERT INTO face_instance (photo_id, face_cluster_id, bounding_box, embedding)
            VALUES (?, ?, '{}', X'00')
            """,
            (photo_id, cluster_id),
        )
        self.conn.commit()

        self.conn.execute("DELETE FROM photo WHERE id = ?", (photo_id,))
        self.conn.commit()

        remaining = self.conn.execute("SELECT COUNT(*) AS c FROM face_instance").fetchone()
        self.assertEqual(remaining["c"], 0)

    def test_photo_location_is_one_row_per_photo(self):
        db.migrate(self.conn)
        photo_id = self._insert_photo()
        loc_id_1 = self.conn.execute(
            "INSERT INTO location_cluster (centroid_lat, centroid_lng) VALUES (37.5, 127.0)"
        ).lastrowid
        loc_id_2 = self.conn.execute(
            "INSERT INTO location_cluster (centroid_lat, centroid_lng) VALUES (37.6, 127.1)"
        ).lastrowid

        self.conn.execute(
            "INSERT INTO photo_location (photo_id, location_cluster_id) VALUES (?, ?)",
            (photo_id, loc_id_1),
        )
        self.conn.commit()

        # A second row for the same photo must fail: photo_id is the PK.
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO photo_location (photo_id, location_cluster_id) VALUES (?, ?)",
                (photo_id, loc_id_2),
            )

    def test_cluster_event_type_and_event_type_check_constraints(self):
        db.migrate(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                """
                INSERT INTO cluster_event (cluster_type, event_type, source_cluster_ids, resulting_cluster_ids)
                VALUES ('face', 'not_a_type', '[1,2]', '[3]')
                """
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                """
                INSERT INTO cluster_event (cluster_type, event_type, source_cluster_ids, resulting_cluster_ids)
                VALUES ('not_a_cluster_type', 'merge', '[1,2]', '[3]')
                """
            )

    def test_autobio_entry_date_is_unique(self):
        db.migrate(self.conn)
        self.conn.execute("INSERT INTO autobio_entry (date) VALUES ('2026-08-24')")
        self.conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO autobio_entry (date) VALUES ('2026-08-24')")


if __name__ == "__main__":
    unittest.main()
