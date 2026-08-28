"""§4.6 Autobio — daily narrative generation, storage, and editing.
Core loop first (per user direction): generate a day's narrative, view
it, edit the text — segment-level photo correction and multi-day combined
narratives are deferred (see README).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import autobio, db, repository, web  # noqa: E402

DIMS = 512


def _embedding(person: int) -> list[float]:
    vector = [0.0] * DIMS
    vector[person % DIMS] = 1.0
    return vector


class AutobioTestCase(unittest.TestCase):
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

    def _photo(self, taken_at, description=None) -> int:
        from PIL import Image

        self._n += 1
        name = f"p{self._n}.jpg"
        path = self.tmp / name
        Image.new("RGB", (200, 200), color=(90, 90, 90)).save(path, "JPEG")
        with self.conn:
            photo_id = repository.insert_photo(
                self.conn,
                picker_media_id=f"pmi-{self._n}",
                taken_at=taken_at,
                original_filename=name,
                original_storage_backend="local",
                original_storage_path=str(path),
            )
            if description:
                repository.set_photo_description(self.conn, photo_id, description)
        return photo_id

    def _name_person(self, photo_id, name, person=0) -> int:
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(person),
            )
            repository.name_cluster(self.conn, "face", cluster_id, name)
        return cluster_id

    def _name_place(self, photo_id, name, lat=37.5, lng=127.0) -> int:
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=lat, lng=lng)
            repository.set_photo_location(self.conn, photo_id=photo_id, location_cluster_id=cluster_id)
            repository.name_cluster(self.conn, "place", cluster_id, name)
        return cluster_id

    def _label_event(self, photo_id, name) -> int:
        with self.conn:
            event_id = repository.get_or_create_event(self.conn, name)
            repository.add_photos_to_event(self.conn, event_id, [photo_id])
        return event_id


class PhotosForDateTests(AutobioTestCase):
    def test_only_photos_on_that_date(self):
        in_range = self._photo("2026-08-14T09:00:00")
        self._photo("2026-08-15T09:00:00")

        photos = repository.photos_for_date(self.conn, "2026-08-14")

        self.assertEqual([p["photo_id"] for p in photos], [in_range])

    def test_ordered_chronologically(self):
        later = self._photo("2026-08-14T18:00:00")
        earlier = self._photo("2026-08-14T09:00:00")

        photos = repository.photos_for_date(self.conn, "2026-08-14")

        self.assertEqual([p["photo_id"] for p in photos], [earlier, later])

    def test_empty_day_returns_empty_list(self):
        self.assertEqual(repository.photos_for_date(self.conn, "2026-01-01"), [])

    def test_includes_named_people_and_places_per_photo(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._name_person(photo_id, "정현")
        self._name_place(photo_id, "코롬방제과점")

        photos = repository.photos_for_date(self.conn, "2026-08-14")

        self.assertEqual(photos[0]["people"], ["정현"])
        self.assertEqual(photos[0]["places"], ["코롬방제과점"])

    def test_includes_event_labels_per_photo(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._label_event(photo_id, "원필 생일")

        photos = repository.photos_for_date(self.conn, "2026-08-14")

        self.assertEqual(photos[0]["events"], ["원필 생일"])

    def test_photo_with_no_event_has_an_empty_events_list(self):
        self._photo("2026-08-14T09:00:00")

        photos = repository.photos_for_date(self.conn, "2026-08-14")

        self.assertEqual(photos[0]["events"], [])

    def test_unnamed_faces_and_places_are_not_included(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0),
            )

        photos = repository.photos_for_date(self.conn, "2026-08-14")

        self.assertEqual(photos[0]["people"], [])

    def test_includes_description(self):
        photo_id = self._photo("2026-08-14T09:00:00", description="A nice lunch")

        photos = repository.photos_for_date(self.conn, "2026-08-14")

        self.assertEqual(photos[0]["description"], "A nice lunch")

    def test_respects_taken_at_override(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            repository.override_taken_at(self.conn, photo_id, "2026-08-20T09:00:00")

        self.assertEqual(repository.photos_for_date(self.conn, "2026-08-14"), [])
        self.assertEqual(len(repository.photos_for_date(self.conn, "2026-08-20")), 1)


class UnlabeledCountTests(AutobioTestCase):
    def test_zero_when_everything_is_named(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._name_person(photo_id, "정현")

        self.assertEqual(repository.count_unlabeled_for_date(self.conn, "2026-08-14"), 0)

    def test_counts_an_unnamed_face_cluster(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0),
            )

        self.assertEqual(repository.count_unlabeled_for_date(self.conn, "2026-08-14"), 1)

    def test_excluded_clusters_do_not_count(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0),
            )
            repository.exclude_cluster(self.conn, "face", cluster_id)

        self.assertEqual(repository.count_unlabeled_for_date(self.conn, "2026-08-14"), 0)

    def test_counts_an_unnamed_place(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            cluster_id = repository.insert_location_cluster(self.conn, lat=37.5, lng=127.0)
            repository.set_photo_location(self.conn, photo_id=photo_id, location_cluster_id=cluster_id)

        self.assertEqual(repository.count_unlabeled_for_date(self.conn, "2026-08-14"), 1)


class AutobioEntryStorageTests(AutobioTestCase):
    def test_nothing_stored_by_default(self):
        self.assertIsNone(repository.get_autobio_entry(self.conn, "2026-08-14"))

    def test_saves_and_reads_back_a_draft(self):
        segments = [{"text": "Had lunch.", "source_photo_ids": [1], "edited": False}]
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=segments, draft_text="Had lunch.",
                has_unlabeled=False,
            )

        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["draft_text"], "Had lunch.")
        self.assertEqual(entry["segments"], segments)
        # final_text starts equal to the draft — nothing to protect yet.
        self.assertEqual(entry["final_text"], "Had lunch.")
        self.assertFalse(entry["is_edited"])
        self.assertFalse(entry["has_unlabeled"])

    def test_stores_the_unlabeled_flag(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="x", has_unlabeled=True
            )

        self.assertTrue(repository.get_autobio_entry(self.conn, "2026-08-14")["has_unlabeled"])

    def test_regenerating_replaces_the_draft_and_final_text_together(self):
        # Nothing was edited yet, so both move together — final_text isn't
        # "protected" until it actually diverges from the draft.
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="First draft.", has_unlabeled=False
            )
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="Second draft.", has_unlabeled=False
            )

        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["draft_text"], "Second draft.")
        self.assertEqual(entry["final_text"], "Second draft.")
        self.assertFalse(entry["is_edited"])
        # One row per date, not a duplicate.
        count = self.conn.execute("SELECT COUNT(*) c FROM autobio_entry").fetchone()["c"]
        self.assertEqual(count, 1)

    def test_regenerating_does_not_clobber_an_existing_edit(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="Draft one.", has_unlabeled=False
            )
            repository.set_autobio_final_text(self.conn, "2026-08-14", "My edited version.")
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="Draft two.", has_unlabeled=False
            )

        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["draft_text"], "Draft two.")
        self.assertEqual(entry["final_text"], "My edited version.")
        self.assertTrue(entry["is_edited"])

    def test_setting_final_text_marks_it_edited(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="Draft.", has_unlabeled=False
            )
            repository.set_autobio_final_text(self.conn, "2026-08-14", "Edited.")

        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertTrue(entry["is_edited"])
        self.assertEqual(entry["final_text"], "Edited.")

    def test_final_text_identical_to_draft_is_not_flagged_as_edited(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="Same.", has_unlabeled=False
            )
            repository.set_autobio_final_text(self.conn, "2026-08-14", "Same.")

        self.assertFalse(repository.get_autobio_entry(self.conn, "2026-08-14")["is_edited"])

    def _seed_two_segments(self):
        segments = [
            {"text": "Morning.", "source_photo_ids": [1], "edited": False},
            {"text": "Evening.", "source_photo_ids": [2], "edited": False},
        ]
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=segments,
                draft_text="Morning.\n\nEvening.", has_unlabeled=False,
            )

    def test_set_segment_text_updates_just_that_segment(self):
        self._seed_two_segments()

        with self.conn:
            repository.set_autobio_segment_text(self.conn, "2026-08-14", 0, "Corrected morning.")

        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["segments"][0]["text"], "Corrected morning.")
        self.assertEqual(entry["segments"][1]["text"], "Evening.")

    def test_set_segment_text_marks_that_segment_edited(self):
        self._seed_two_segments()

        with self.conn:
            repository.set_autobio_segment_text(self.conn, "2026-08-14", 0, "Corrected morning.")

        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertTrue(entry["segments"][0]["edited"])
        self.assertFalse(entry["segments"][1]["edited"])

    def test_set_segment_text_reassembles_final_text(self):
        self._seed_two_segments()

        with self.conn:
            repository.set_autobio_segment_text(self.conn, "2026-08-14", 0, "Corrected morning.")

        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["final_text"], "Corrected morning.\n\nEvening.")
        self.assertTrue(entry["is_edited"])

    def test_set_segment_text_can_mark_not_edited(self):
        # regenerate_segment uses this — a fresh LLM draft isn't a user edit.
        self._seed_two_segments()

        with self.conn:
            repository.set_autobio_segment_text(self.conn, "2026-08-14", 0, "Regenerated.", edited=False)

        self.assertFalse(repository.get_autobio_entry(self.conn, "2026-08-14")["segments"][0]["edited"])

    def test_out_of_range_index_raises(self):
        self._seed_two_segments()

        with self.conn:
            with self.assertRaises(IndexError):
                repository.set_autobio_segment_text(self.conn, "2026-08-14", 5, "x")

    def test_missing_entry_raises(self):
        with self.assertRaises(ValueError):
            repository.set_autobio_segment_text(self.conn, "2099-01-01", 0, "x")

    def test_list_entries_newest_first(self):
        with self.conn:
            repository.save_autobio_draft(self.conn, date="2026-08-10", segments=[], draft_text="a", has_unlabeled=False)
            repository.save_autobio_draft(self.conn, date="2026-08-20", segments=[], draft_text="b", has_unlabeled=False)
            repository.save_autobio_draft(self.conn, date="2026-08-15", segments=[], draft_text="c", has_unlabeled=False)

        dates = [e["date"] for e in repository.list_autobio_entries(self.conn)]
        self.assertEqual(dates, ["2026-08-20", "2026-08-15", "2026-08-10"])


class BuildPromptTests(unittest.TestCase):
    def test_includes_time_people_places_and_description(self):
        photos = [
            {
                "photo_id": 1,
                "taken_at": "2026-08-14T09:12:00",
                "people": ["정현"],
                "places": ["코롬방제과점"],
                "description": "great pastries",
            }
        ]

        prompt = autobio.build_prompt(photos)

        self.assertIn("[id 1]", prompt)
        self.assertIn("09:12", prompt)
        self.assertIn("정현", prompt)
        self.assertIn("코롬방제과점", prompt)
        self.assertIn("great pastries", prompt)

    def test_includes_event_label(self):
        photos = [
            {
                "photo_id": 1,
                "taken_at": "2026-08-15T13:00:00",
                "people": [],
                "places": [],
                "events": ["원필 생일"],
            }
        ]

        prompt = autobio.build_prompt(photos)

        self.assertIn("원필 생일", prompt)

    def test_photo_with_no_extra_data_still_appears(self):
        photos = [{"photo_id": 5, "taken_at": "2026-08-14T10:00:00", "people": [], "places": [], "description": None}]

        prompt = autobio.build_prompt(photos)

        self.assertIn("[id 5]", prompt)


class ParseLlmResponseTests(unittest.TestCase):
    def test_parses_well_formed_json(self):
        raw = json.dumps({"segments": [{"text": "Had lunch.", "source_photo_ids": [1, 2]}]})

        segments = autobio._parse_llm_response(raw, valid_photo_ids={1, 2})

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "Had lunch.")
        self.assertEqual(segments[0].source_photo_ids, [1, 2])

    def test_strips_markdown_code_fence(self):
        raw = "```json\n" + json.dumps({"segments": [{"text": "x", "source_photo_ids": [1]}]}) + "\n```"

        segments = autobio._parse_llm_response(raw, valid_photo_ids={1})

        self.assertEqual(segments[0].text, "x")

    def test_drops_hallucinated_photo_ids(self):
        raw = json.dumps({"segments": [{"text": "x", "source_photo_ids": [1, 999]}]})

        segments = autobio._parse_llm_response(raw, valid_photo_ids={1})

        self.assertEqual(segments[0].source_photo_ids, [1])

    def test_drops_empty_text_segments(self):
        raw = json.dumps(
            {"segments": [{"text": "  ", "source_photo_ids": [1]}, {"text": "real", "source_photo_ids": [1]}]}
        )

        segments = autobio._parse_llm_response(raw, valid_photo_ids={1})

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "real")

    def test_invalid_json_raises_a_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            autobio._parse_llm_response("not json at all", valid_photo_ids={1})
        self.assertIn("not json at all", str(ctx.exception))

    def test_missing_segments_key_raises(self):
        with self.assertRaises(ValueError):
            autobio._parse_llm_response(json.dumps({"oops": []}), valid_photo_ids={1})

    def test_empty_segments_list_raises(self):
        with self.assertRaises(ValueError):
            autobio._parse_llm_response(json.dumps({"segments": []}), valid_photo_ids={1})

    def test_all_segments_empty_after_filtering_raises(self):
        raw = json.dumps({"segments": [{"text": "  ", "source_photo_ids": [1]}]})
        with self.assertRaises(ValueError):
            autobio._parse_llm_response(raw, valid_photo_ids={1})


class GenerateDailyEntryTests(AutobioTestCase):
    def test_raises_when_no_photos_for_that_date(self):
        with self.assertRaises(autobio.NoPhotosForDate):
            autobio.generate_daily_entry(self.conn, "2026-01-01", complete=lambda *a, **kw: "")

    def test_generates_and_stores_an_entry(self):
        self._photo("2026-08-14T09:00:00")

        def fake_complete(prompt, system=None):
            return json.dumps({"segments": [{"text": "A nice day.", "source_photo_ids": [1]}]})

        entry = autobio.generate_daily_entry(self.conn, "2026-08-14", complete=fake_complete)

        self.assertEqual(entry["draft_text"], "A nice day.")
        self.assertEqual(entry["final_text"], "A nice day.")
        stored = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(stored["draft_text"], "A nice day.")

    def test_multiple_segments_are_joined_with_blank_lines(self):
        self._photo("2026-08-14T09:00:00")

        def fake_complete(prompt, system=None):
            return json.dumps(
                {"segments": [{"text": "Morning.", "source_photo_ids": [1]}, {"text": "Evening.", "source_photo_ids": [1]}]}
            )

        entry = autobio.generate_daily_entry(self.conn, "2026-08-14", complete=fake_complete)

        self.assertEqual(entry["draft_text"], "Morning.\n\nEvening.")

    def test_records_the_unlabeled_flag(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0),
            )

        def fake_complete(prompt, system=None):
            return json.dumps({"segments": [{"text": "x", "source_photo_ids": [photo_id]}]})

        entry = autobio.generate_daily_entry(self.conn, "2026-08-14", complete=fake_complete)

        self.assertTrue(entry["has_unlabeled"])

    def test_regenerating_preserves_an_existing_edit(self):
        self._photo("2026-08-14T09:00:00")

        def fake_complete(prompt, system=None):
            return json.dumps({"segments": [{"text": "Original.", "source_photo_ids": [1]}]})

        autobio.generate_daily_entry(self.conn, "2026-08-14", complete=fake_complete)
        with self.conn:
            repository.set_autobio_final_text(self.conn, "2026-08-14", "My own version.")

        def fake_complete_2(prompt, system=None):
            return json.dumps({"segments": [{"text": "Regenerated.", "source_photo_ids": [1]}]})

        entry = autobio.generate_daily_entry(self.conn, "2026-08-14", complete=fake_complete_2)

        self.assertEqual(entry["draft_text"], "Regenerated.")
        self.assertEqual(entry["final_text"], "My own version.")

    def test_prompt_receives_the_days_photo_data(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._name_person(photo_id, "정현")
        self._label_event(photo_id, "원필 생일")
        captured = {}

        def fake_complete(prompt, system=None):
            captured["prompt"] = prompt
            captured["system"] = system
            return json.dumps({"segments": [{"text": "x", "source_photo_ids": [photo_id]}]})

        autobio.generate_daily_entry(self.conn, "2026-08-14", complete=fake_complete)

        self.assertIn("정현", captured["prompt"])
        self.assertIn("원필 생일", captured["prompt"])
        self.assertIn(f"[id {photo_id}]", captured["prompt"])
        self.assertEqual(captured["system"], autobio.SYSTEM_PROMPT)

    def test_prompt_carries_an_explicit_language_directive(self):
        # Real bug this replaces: the old prompt only said "write in
        # whichever language the given data is mostly in", which flipped
        # day to day depending on what happened to be in the photo
        # metadata. Now it's an explicit, stable per-call choice.
        photo_id = self._photo("2026-08-14T09:00:00")
        captured = {}

        def fake_complete(prompt, system=None):
            captured["prompt"] = prompt
            return json.dumps({"segments": [{"text": "x", "source_photo_ids": [photo_id]}]})

        autobio.generate_daily_entry(self.conn, "2026-08-14", complete=fake_complete, language="fr")

        self.assertIn("Write the entire entry in French", captured["prompt"])

    def test_default_language_is_korean(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        captured = {}

        def fake_complete(prompt, system=None):
            captured["prompt"] = prompt
            return json.dumps({"segments": [{"text": "x", "source_photo_ids": [photo_id]}]})

        autobio.generate_daily_entry(self.conn, "2026-08-14", complete=fake_complete)

        self.assertIn("Write the entire entry in Korean", captured["prompt"])


class RegenerateSegmentTests(AutobioTestCase):
    """§4.6's per-segment correction flow: fix a photo's metadata, then
    regenerate just the one segment built from it — not the whole day."""

    def _seed_two_segment_entry(self, photo_id_a, photo_id_b):
        segments = [
            {"text": "Morning.", "source_photo_ids": [photo_id_a], "edited": False},
            {"text": "Evening.", "source_photo_ids": [photo_id_b], "edited": False},
        ]
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=segments,
                draft_text="Morning.\n\nEvening.", has_unlabeled=False,
            )

    def test_raises_for_a_date_with_no_entry(self):
        with self.assertRaises(autobio.NoPhotosForDate):
            autobio.regenerate_segment(self.conn, "2026-08-14", 0, complete=lambda *a, **kw: "")

    def test_raises_for_an_out_of_range_index(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)

        with self.assertRaises(autobio.NoSuchSegment):
            autobio.regenerate_segment(self.conn, "2026-08-14", 9, complete=lambda *a, **kw: "")

    def test_regenerates_only_the_target_segment(self):
        photo_a = self._photo("2026-08-14T09:00:00")
        photo_b = self._photo("2026-08-14T19:00:00")
        self._seed_two_segment_entry(photo_a, photo_b)

        def fake_complete(prompt, system=None):
            return json.dumps({"segments": [{"text": "New morning text.", "source_photo_ids": [photo_a]}]})

        entry = autobio.regenerate_segment(self.conn, "2026-08-14", 0, complete=fake_complete)

        self.assertEqual(entry["segments"][0]["text"], "New morning text.")
        self.assertEqual(entry["segments"][1]["text"], "Evening.")  # untouched

    def test_prompt_is_scoped_to_just_that_segments_photos(self):
        photo_a = self._photo("2026-08-14T09:00:00")
        photo_b = self._photo("2026-08-14T19:00:00")
        self._name_person(photo_a, "정현")
        self._seed_two_segment_entry(photo_a, photo_b)
        captured = {}

        def fake_complete(prompt, system=None):
            captured["prompt"] = prompt
            return json.dumps({"segments": [{"text": "x", "source_photo_ids": [photo_a]}]})

        autobio.regenerate_segment(self.conn, "2026-08-14", 0, complete=fake_complete)

        self.assertIn(f"[id {photo_a}]", captured["prompt"])
        self.assertNotIn(f"[id {photo_b}]", captured["prompt"])

    def test_picks_up_a_metadata_correction_made_since_generation(self):
        # The whole point: fix a photo's name/place, then regenerate the
        # segment built from it, and see the fix reflected.
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)
        captured = {}

        def fake_complete(prompt, system=None):
            captured["prompt"] = prompt
            return json.dumps({"segments": [{"text": "x", "source_photo_ids": [photo_id]}]})

        # Correction made *after* the entry was first generated.
        self._name_person(photo_id, "정현")

        autobio.regenerate_segment(self.conn, "2026-08-14", 0, complete=fake_complete)

        self.assertIn("정현", captured["prompt"])

    def test_regenerated_segment_is_not_marked_as_a_user_edit(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)

        def fake_complete(prompt, system=None):
            return json.dumps({"segments": [{"text": "x", "source_photo_ids": [photo_id]}]})

        entry = autobio.regenerate_segment(self.conn, "2026-08-14", 0, complete=fake_complete)

        self.assertFalse(entry["segments"][0]["edited"])

    def test_reassembles_final_text(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)

        def fake_complete(prompt, system=None):
            return json.dumps({"segments": [{"text": "New morning.", "source_photo_ids": [photo_id]}]})

        entry = autobio.regenerate_segment(self.conn, "2026-08-14", 0, complete=fake_complete)

        self.assertEqual(entry["final_text"], "New morning.\n\nEvening.")

    def test_raises_a_clear_error_if_segments_photos_were_deleted(self):
        # Rather than hitting the LLM with an empty photo list.
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14",
                segments=[{"text": "x", "source_photo_ids": [99999], "edited": False}],
                draft_text="x", has_unlabeled=False,
            )

        with self.assertRaises(ValueError):
            autobio.regenerate_segment(self.conn, "2026-08-14", 0, complete=lambda *a, **kw: "")


class AutobioRoutesTests(AutobioTestCase):
    def test_index_renders_with_no_entries(self):
        # /autobio is the combined-narrative-only tab now; daily entries
        # live on /autobio/daily (the "Diary" tab) — see AutobioDailyIndexTests.
        response = self.client.get("/autobio/daily")
        self.assertEqual(response.status_code, 200)
        # ui_language defaults to Korean — see test_i18n.py for translation coverage.
        self.assertIn("아직 일기 항목이 없습니다".encode(), response.data)

    def test_index_lists_entries(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="A day.", has_unlabeled=False
            )

        response = self.client.get("/autobio/daily")

        self.assertIn(b"2026-08-14", response.data)
        self.assertIn(b"A day.", response.data)

    def test_generate_requires_a_date(self):
        response = self.client.post("/autobio/generate", data={})
        self.assertEqual(response.status_code, 400)

    def test_generate_reports_no_photos_not_a_crash(self):
        response = self.client.post("/autobio/generate", data={"date": "2026-01-01"})
        self.assertEqual(response.status_code, 400)
        # error.html's message comes straight from autobio.NoPhotosForDate's
        # plain-English text, not from i18n.py — that route isn't translated.
        self.assertIn(b"No photos found", response.data)

    @patch("photo_grouping.web.llm.complete")
    def test_generate_happy_path_redirects_to_the_entry(self, mock_complete):
        import json

        self._photo("2026-08-14T09:00:00")
        mock_complete.return_value = json.dumps({"segments": [{"text": "A day out.", "source_photo_ids": [1]}]})

        response = self.client.post("/autobio/generate", data={"date": "2026-08-14"})

        self.assertEqual(response.status_code, 302)
        self.assertIn("/autobio/2026-08-14", response.headers["Location"])
        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["draft_text"], "A day out.")

    @patch("photo_grouping.web.llm.complete")
    def test_generate_reports_llm_not_configured(self, mock_complete):
        from photo_grouping import llm

        self._photo("2026-08-14T09:00:00")
        mock_complete.side_effect = llm.LLMNotConfigured("No key found.")

        response = self.client.post("/autobio/generate", data={"date": "2026-08-14"})

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No key found", response.data)

    @patch("photo_grouping.web.llm.complete")
    def test_generate_reports_a_bad_model_response_not_a_crash(self, mock_complete):
        self._photo("2026-08-14T09:00:00")
        mock_complete.return_value = "not json"

        response = self.client.post("/autobio/generate", data={"date": "2026-08-14"})

        self.assertEqual(response.status_code, 502)

    def test_entry_page_offers_to_generate_when_photos_exist_but_no_entry_yet(self):
        self._photo("2026-08-14T09:00:00")

        response = self.client.get("/autobio/2026-08-14")

        self.assertEqual(response.status_code, 200)
        self.assertIn("이야기 생성".encode(), response.data)  # ui_language defaults to Korean

    def test_entry_page_reports_no_photos(self):
        response = self.client.get("/autobio/2026-01-01")

        self.assertEqual(response.status_code, 200)
        self.assertIn("사진이 없습니다".encode(), response.data)  # ui_language defaults to Korean

    def test_entry_page_shows_the_saved_text(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14",
                segments=[{"text": "Draft text.", "source_photo_ids": [], "edited": False}],
                draft_text="Draft text.", has_unlabeled=False,
            )

        response = self.client.get("/autobio/2026-08-14")

        self.assertIn(b"Draft text.", response.data)

    def test_entry_page_shows_the_live_unlabeled_count_not_the_stale_snapshot(self):
        # The stored has_unlabeled flag is a generation-time snapshot;
        # the page itself should reflect labeling done since then.
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0),
            )
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="x", has_unlabeled=True
            )
            # Now label it — the flag on the stored row is still True, but
            # the live count should be 0.
            repository.name_cluster(self.conn, "face", cluster_id, "정현")

        response = self.client.get("/autobio/2026-08-14")

        self.assertNotIn(b"unlabeled", response.data.lower())

    def test_save_route_updates_final_text_and_redirects(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="Draft.", has_unlabeled=False
            )

        response = self.client.post(
            "/autobio/2026-08-14/save", data={"final_text": "My final version."}
        )

        self.assertEqual(response.status_code, 302)
        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["final_text"], "My final version.")
        self.assertTrue(entry["is_edited"])

    def test_nav_link_present(self):
        response = self.client.get("/")
        self.assertIn(b'href="/autobio"', response.data)

    def test_delete_route_removes_the_entry_and_redirects(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="Draft.", has_unlabeled=False
            )

        response = self.client.post("/autobio/2026-08-14/delete")

        self.assertEqual(response.status_code, 302)
        self.assertIsNone(repository.get_autobio_entry(self.conn, "2026-08-14"))

    def test_deleting_a_nonexistent_entry_is_a_harmless_no_op(self):
        response = self.client.post("/autobio/2026-08-14/delete")

        self.assertEqual(response.status_code, 302)


class AutobioSegmentRoutesTests(AutobioTestCase):
    def _seed_two_segment_entry(self, photo_id_a, photo_id_b):
        segments = [
            {"text": "Morning.", "source_photo_ids": [photo_id_a], "edited": False},
            {"text": "Evening.", "source_photo_ids": [photo_id_b], "edited": False},
        ]
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=segments,
                draft_text="Morning.\n\nEvening.", has_unlabeled=False,
            )

    def test_segment_save_updates_and_redirects(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)

        response = self.client.post(
            "/autobio/2026-08-14/segment/0/save", data={"text": "Corrected."}
        )

        self.assertEqual(response.status_code, 302)
        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["segments"][0]["text"], "Corrected.")
        self.assertTrue(entry["segments"][0]["edited"])

    def test_segment_save_out_of_range_is_404(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)

        response = self.client.post(
            "/autobio/2026-08-14/segment/9/save", data={"text": "x"}
        )

        self.assertEqual(response.status_code, 404)

    def test_segment_save_for_nonexistent_entry_is_404(self):
        response = self.client.post(
            "/autobio/2099-01-01/segment/0/save", data={"text": "x"}
        )
        self.assertEqual(response.status_code, 404)

    @patch("photo_grouping.web.llm.complete")
    def test_segment_regenerate_happy_path(self, mock_complete):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)
        mock_complete.return_value = json.dumps(
            {"segments": [{"text": "New text.", "source_photo_ids": [photo_id]}]}
        )

        response = self.client.post("/autobio/2026-08-14/segment/0/regenerate")

        self.assertEqual(response.status_code, 302)
        entry = repository.get_autobio_entry(self.conn, "2026-08-14")
        self.assertEqual(entry["segments"][0]["text"], "New text.")

    def test_segment_regenerate_out_of_range_is_404(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)

        response = self.client.post("/autobio/2026-08-14/segment/9/regenerate")

        self.assertEqual(response.status_code, 404)

    def test_segment_regenerate_for_nonexistent_entry_is_404(self):
        response = self.client.post("/autobio/2099-01-01/segment/0/regenerate")
        self.assertEqual(response.status_code, 404)

    @patch("photo_grouping.web.llm.complete")
    def test_segment_regenerate_reports_llm_not_configured(self, mock_complete):
        from photo_grouping import llm

        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)
        mock_complete.side_effect = llm.LLMNotConfigured("No key found.")

        response = self.client.post("/autobio/2026-08-14/segment/0/regenerate")

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No key found", response.data)

    @patch("photo_grouping.web.llm.complete")
    def test_segment_regenerate_reports_bad_model_response(self, mock_complete):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)
        mock_complete.return_value = "not json"

        response = self.client.post("/autobio/2026-08-14/segment/0/regenerate")

        self.assertEqual(response.status_code, 502)

    def test_entry_page_renders_segment_editors_and_stepper(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)

        response = self.client.get("/autobio/2026-08-14")

        self.assertIn(b"Morning.", response.data)
        self.assertIn(b"Evening.", response.data)
        self.assertIn(b"/autobio/2026-08-14/segment/0/save", response.data)
        self.assertIn(b"/autobio/2026-08-14/segment/0/regenerate", response.data)
        self.assertIn(f"/photo/{photo_id}".encode(), response.data)

    def test_edited_segment_shows_badge(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        self._seed_two_segment_entry(photo_id, photo_id)
        with self.conn:
            repository.set_autobio_segment_text(self.conn, "2026-08-14", 0, "Edited text.")

        response = self.client.get("/autobio/2026-08-14")

        self.assertIn(b"Edited text.", response.data)
        self.assertIn(b"badge", response.data)


class AutobioExportRoutesTests(AutobioTestCase):
    def _seed_entry(self):
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="A day out.", has_unlabeled=False
            )

    def test_md_export(self):
        self._seed_entry()

        response = self.client.get("/autobio/2026-08-14/export/md")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/markdown")
        self.assertIn(b"A day out.", response.data)
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))

    def test_txt_export(self):
        self._seed_entry()

        response = self.client.get("/autobio/2026-08-14/export/txt")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/plain")

    def test_docx_export(self):
        self._seed_entry()

        response = self.client.get("/autobio/2026-08-14/export/docx")

        self.assertEqual(response.status_code, 200)
        self.assertIn("wordprocessingml", response.mimetype)

    def test_unknown_format_is_404(self):
        self._seed_entry()

        response = self.client.get("/autobio/2026-08-14/export/pdf")

        self.assertEqual(response.status_code, 404)

    def test_export_of_a_nonexistent_entry_is_404(self):
        response = self.client.get("/autobio/2026-01-01/export/md")
        self.assertEqual(response.status_code, 404)

    def test_export_reflects_a_saved_edit(self):
        self._seed_entry()
        self.client.post("/autobio/2026-08-14/save", data={"final_text": "My own words."})

        response = self.client.get("/autobio/2026-08-14/export/txt")

        self.assertIn(b"My own words.", response.data)
        self.assertNotIn(b"A day out.", response.data)

    def test_entry_page_has_export_links(self):
        self._seed_entry()

        response = self.client.get("/autobio/2026-08-14")

        self.assertIn(b"/autobio/2026-08-14/export/md", response.data)
        self.assertIn(b"/autobio/2026-08-14/export/txt", response.data)
        self.assertIn(b"/autobio/2026-08-14/export/docx", response.data)


class AutobioSummaryStorageTests(AutobioTestCase):
    def test_nothing_stored_by_default(self):
        self.assertIsNone(repository.get_autobio_summary(self.conn, "2026-08-10", "2026-08-14"))

    def test_saves_and_reads_back(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[1, 2], text="A great week.",
            )

        summary = repository.get_autobio_summary(self.conn, "2026-08-10", "2026-08-14")
        self.assertEqual(summary["text"], "A great week.")
        self.assertEqual(summary["source_entry_ids"], [1, 2])

    def test_regenerating_the_same_range_replaces_it_not_duplicates(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[1], text="First version.",
            )
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[1, 2], text="Second version.",
            )

        summary = repository.get_autobio_summary(self.conn, "2026-08-10", "2026-08-14")
        self.assertEqual(summary["text"], "Second version.")
        count = self.conn.execute("SELECT COUNT(*) c FROM autobio_summary").fetchone()["c"]
        self.assertEqual(count, 1)

    def test_set_text_updates_in_place(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[], text="Draft.",
            )
            repository.set_autobio_summary_text(self.conn, "2026-08-10", "2026-08-14", "Edited.")

        self.assertEqual(
            repository.get_autobio_summary(self.conn, "2026-08-10", "2026-08-14")["text"], "Edited."
        )

    def test_list_newest_start_date_first(self):
        with self.conn:
            repository.save_autobio_summary(self.conn, start_date="2026-08-01", end_date="2026-08-03", source_entry_ids=[], text="a")
            repository.save_autobio_summary(self.conn, start_date="2026-08-10", end_date="2026-08-12", source_entry_ids=[], text="b")

        starts = [s["start_date"] for s in repository.list_autobio_summaries(self.conn)]
        self.assertEqual(starts, ["2026-08-10", "2026-08-01"])


class AutobioSettingsTests(AutobioTestCase):
    def test_on_by_default(self):
        self.assertTrue(repository.get_autobio_settings(self.conn)["show_unlabeled_nudge"])

    def test_can_be_turned_off(self):
        with self.conn:
            repository.set_autobio_show_unlabeled_nudge(self.conn, False)
        self.assertFalse(repository.get_autobio_settings(self.conn)["show_unlabeled_nudge"])

    def test_can_be_turned_back_on(self):
        with self.conn:
            repository.set_autobio_show_unlabeled_nudge(self.conn, False)
            repository.set_autobio_show_unlabeled_nudge(self.conn, True)
        self.assertTrue(repository.get_autobio_settings(self.conn)["show_unlabeled_nudge"])

    def test_settings_route_turns_it_off(self):
        response = self.client.post("/autobio/settings", data={})  # unchecked box sends nothing

        self.assertEqual(response.status_code, 302)
        self.assertFalse(repository.get_autobio_settings(self.conn)["show_unlabeled_nudge"])

    def test_settings_route_turns_it_on(self):
        with self.conn:
            repository.set_autobio_show_unlabeled_nudge(self.conn, False)

        self.client.post("/autobio/settings", data={"show_unlabeled_nudge": "on"})

        self.assertTrue(repository.get_autobio_settings(self.conn)["show_unlabeled_nudge"])

    def test_index_shows_the_checkbox_checked_by_default(self):
        # Settings live on the Diary tab (/autobio/daily) now.
        response = self.client.get("/autobio/daily")
        self.assertIn(b'id="show-unlabeled-nudge"', response.data)
        self.assertIn(b"checked", response.data)

    def test_nudge_hidden_on_entry_page_when_setting_is_off(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0),
            )
            repository.save_autobio_draft(
                self.conn, date="2026-08-14",
                segments=[{"text": "x", "source_photo_ids": [photo_id], "edited": False}],
                draft_text="x", has_unlabeled=True,
            )
            repository.set_autobio_show_unlabeled_nudge(self.conn, False)

        response = self.client.get("/autobio/2026-08-14")

        self.assertNotIn(b"unlabeled", response.data.lower())

    def test_nudge_shown_on_entry_page_when_setting_is_on(self):
        photo_id = self._photo("2026-08-14T09:00:00")
        with self.conn:
            cluster_id = repository.insert_face_cluster(self.conn)
            repository.insert_face_instance(
                self.conn, photo_id=photo_id, face_cluster_id=cluster_id,
                bounding_box={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                embedding=_embedding(0),
            )
            repository.save_autobio_draft(
                self.conn, date="2026-08-14",
                segments=[{"text": "x", "source_photo_ids": [photo_id], "edited": False}],
                draft_text="x", has_unlabeled=True,
            )

        response = self.client.get("/autobio/2026-08-14")

        self.assertIn("이름 없는".encode(), response.data)  # ui_language defaults to Korean


class GenerateCombinedNarrativeTests(AutobioTestCase):
    def test_raises_when_no_photos_anywhere_in_range(self):
        with self.assertRaises(autobio.NoEntriesForRange):
            autobio.generate_combined_narrative(
                self.conn, "2026-01-01", "2026-01-03", complete=lambda *a, **kw: ""
            )

    def test_end_before_start_raises(self):
        with self.assertRaises(ValueError):
            autobio.generate_combined_narrative(
                self.conn, "2026-08-14", "2026-08-10", complete=lambda *a, **kw: ""
            )

    def test_generates_missing_daily_entries_then_combines_them(self):
        self._photo("2026-08-10T09:00:00")
        self._photo("2026-08-12T09:00:00")
        calls = []

        def fake_complete(prompt, system=None):
            calls.append(system)
            if system == autobio.SYSTEM_PROMPT:
                return json.dumps({"segments": [{"text": "Day text.", "source_photo_ids": [1]}]})
            return "Combined narrative text."

        summary = autobio.generate_combined_narrative(
            self.conn, "2026-08-10", "2026-08-14", complete=fake_complete
        )

        self.assertEqual(summary["text"], "Combined narrative text.")
        # Both days generated a daily entry as a side effect.
        self.assertIsNotNone(repository.get_autobio_entry(self.conn, "2026-08-10"))
        self.assertIsNotNone(repository.get_autobio_entry(self.conn, "2026-08-12"))
        # One daily-generation call per day-with-photos, plus the combine call.
        self.assertEqual(calls.count(autobio.SYSTEM_PROMPT), 2)
        self.assertEqual(calls.count(autobio.COMBINE_SYSTEM_PROMPT), 1)

    def test_days_with_no_photos_are_silently_skipped(self):
        self._photo("2026-08-10T09:00:00")
        # 08-11 through 08-14 have no photos at all.

        def fake_complete(prompt, system=None):
            if system == autobio.SYSTEM_PROMPT:
                return json.dumps({"segments": [{"text": "x", "source_photo_ids": [1]}]})
            return "Combined."

        summary = autobio.generate_combined_narrative(
            self.conn, "2026-08-10", "2026-08-14", complete=fake_complete
        )

        self.assertEqual(summary["source_entry_ids"], [repository.get_autobio_entry(self.conn, "2026-08-10")["id"]])

    def test_reuses_an_existing_daily_entry_without_regenerating_it(self):
        self._photo("2026-08-10T09:00:00")
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-10", segments=[], draft_text="Draft.", has_unlabeled=False
            )
            repository.set_autobio_final_text(self.conn, "2026-08-10", "My own edit.")

        def fake_complete(prompt, system=None):
            # Should never be asked to draft 08-10 — an entry already exists.
            self.assertNotEqual(system, autobio.SYSTEM_PROMPT)
            self.assertIn("My own edit.", prompt)
            return "Combined."

        autobio.generate_combined_narrative(self.conn, "2026-08-10", "2026-08-10", complete=fake_complete)

        # The existing edit must be untouched by the combine.
        self.assertEqual(repository.get_autobio_entry(self.conn, "2026-08-10")["final_text"], "My own edit.")

    def test_regenerating_overwrites_the_summary_text(self):
        self._photo("2026-08-10T09:00:00")

        def fake_complete(prompt, system=None):
            if system == autobio.SYSTEM_PROMPT:
                return json.dumps({"segments": [{"text": "x", "source_photo_ids": [1]}]})
            return "First combine."

        autobio.generate_combined_narrative(self.conn, "2026-08-10", "2026-08-10", complete=fake_complete)

        def fake_complete_2(prompt, system=None):
            if system == autobio.SYSTEM_PROMPT:
                return json.dumps({"segments": [{"text": "x", "source_photo_ids": [1]}]})
            return "Second combine."

        summary = autobio.generate_combined_narrative(
            self.conn, "2026-08-10", "2026-08-10", complete=fake_complete_2
        )

        self.assertEqual(summary["text"], "Second combine.")

    def test_language_is_threaded_through_to_both_daily_and_combine_calls(self):
        self._photo("2026-08-10T09:00:00")
        captured = {}

        def fake_complete(prompt, system=None):
            if system == autobio.SYSTEM_PROMPT:
                captured["daily_prompt"] = prompt
                return json.dumps({"segments": [{"text": "x", "source_photo_ids": [1]}]})
            captured["combine_prompt"] = prompt
            return "Combined."

        autobio.generate_combined_narrative(
            self.conn, "2026-08-10", "2026-08-10", complete=fake_complete, language="uk"
        )

        self.assertIn("Write the entire entry in Ukrainian", captured["daily_prompt"])
        self.assertIn("Write the combined narrative in Ukrainian", captured["combine_prompt"])


class AutobioSummaryRoutesTests(AutobioTestCase):
    def test_generate_range_requires_both_dates(self):
        response = self.client.post("/autobio/generate-range", data={"start_date": "2026-08-10"})
        self.assertEqual(response.status_code, 400)

    def test_generate_range_reports_no_photos_not_a_crash(self):
        response = self.client.post(
            "/autobio/generate-range", data={"start_date": "2026-01-01", "end_date": "2026-01-03"}
        )
        self.assertEqual(response.status_code, 400)
        # error.html's message comes straight from autobio.NoEntriesForRange's
        # plain-English text, not from i18n.py — that route isn't translated.
        self.assertIn(b"No photos found", response.data)

    def test_generate_range_reports_end_before_start(self):
        response = self.client.post(
            "/autobio/generate-range", data={"start_date": "2026-08-14", "end_date": "2026-08-10"}
        )
        self.assertEqual(response.status_code, 400)

    @patch("photo_grouping.web.llm.complete")
    def test_generate_range_happy_path_redirects(self, mock_complete):
        from photo_grouping import autobio

        self._photo("2026-08-10T09:00:00")

        def fake_complete(prompt, system=None):
            if system == autobio.SYSTEM_PROMPT:
                return json.dumps({"segments": [{"text": "x", "source_photo_ids": [1]}]})
            return "Combined text."

        mock_complete.side_effect = fake_complete

        response = self.client.post(
            "/autobio/generate-range", data={"start_date": "2026-08-10", "end_date": "2026-08-10"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/autobio/summary/2026-08-10/2026-08-10", response.headers["Location"])

    def test_summary_view_offers_to_generate_when_not_yet_generated(self):
        response = self.client.get("/autobio/summary/2026-08-10/2026-08-14")

        self.assertEqual(response.status_code, 200)
        self.assertIn("합쳐진 이야기 생성".encode(), response.data)  # ui_language defaults to Korean

    def test_summary_view_shows_saved_text(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[], text="A great week.",
            )

        response = self.client.get("/autobio/summary/2026-08-10/2026-08-14")

        self.assertIn(b"A great week.", response.data)

    def test_summary_save_route(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[], text="Draft.",
            )

        response = self.client.post(
            "/autobio/summary/2026-08-10/2026-08-14/save", data={"text": "My edit."}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            repository.get_autobio_summary(self.conn, "2026-08-10", "2026-08-14")["text"], "My edit."
        )

    def test_summary_delete_route_removes_it_and_redirects(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[], text="Draft.",
            )

        response = self.client.post("/autobio/summary/2026-08-10/2026-08-14/delete")

        self.assertEqual(response.status_code, 302)
        self.assertIsNone(repository.get_autobio_summary(self.conn, "2026-08-10", "2026-08-14"))

    def test_summary_export_md(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[], text="A great week.",
            )

        response = self.client.get("/autobio/summary/2026-08-10/2026-08-14/export/md")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"A great week.", response.data)

    def test_summary_export_of_nonexistent_summary_is_404(self):
        response = self.client.get("/autobio/summary/2026-08-10/2026-08-14/export/md")
        self.assertEqual(response.status_code, 404)

    def test_index_lists_summaries(self):
        with self.conn:
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[], text="A great week.",
            )

        response = self.client.get("/autobio")

        self.assertIn(b"2026-08-10", response.data)
        self.assertIn(b"A great week.", response.data)

    def test_index_links_summary_to_its_composing_daily_entries(self):
        with self.conn:
            entry_id = repository.save_autobio_draft(
                self.conn, date="2026-08-12", segments=[], draft_text="x", has_unlabeled=False
            )
            repository.save_autobio_summary(
                self.conn, start_date="2026-08-10", end_date="2026-08-14",
                source_entry_ids=[entry_id], text="A great week.",
            )

        response = self.client.get("/autobio")

        self.assertIn(b"/autobio/2026-08-12", response.data)


class LLMGatingTests(AutobioTestCase):
    """Diary/Autobio show guidance toward /settings/connect instead of a
    generate form when no AI provider is configured (see web.py's
    _llm_configured()) — requested directly after the "Connect your
    accounts" feature shipped. Real ANTHROPIC_KEY_PATH/OPENAI_KEY_PATH are
    explicitly patched to tmp paths in every test here rather than left at
    their real repo/secrets/ defaults — otherwise these tests would pass
    or fail based on whatever happens to be configured on the machine
    running them, not on the behavior actually being tested."""

    def setUp(self):
        super().setUp()
        self.anthropic_path = self.tmp / "secrets" / "anthropic_api_key.txt"
        self.openai_path = self.tmp / "secrets" / "openai_api_key.txt"
        self._patches = [
            patch("photo_grouping.web.ANTHROPIC_KEY_PATH", self.anthropic_path),
            patch("photo_grouping.web.OPENAI_KEY_PATH", self.openai_path),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        super().tearDown()

    def test_diary_shows_guidance_when_nothing_configured(self):
        response = self.client.get("/autobio/daily")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/settings/connect", response.data)
        self.assertNotIn(b'name="date"', response.data)  # the generate form is hidden

    def test_diary_shows_the_generate_form_once_a_key_exists(self):
        self.anthropic_path.parent.mkdir(parents=True)
        self.anthropic_path.write_text("sk-ant-fake\n")

        response = self.client.get("/autobio/daily")

        self.assertIn(b'name="date"', response.data)
        self.assertNotIn(b"/settings/connect", response.data)

    def test_autobio_shows_guidance_when_nothing_configured(self):
        response = self.client.get("/autobio")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/settings/connect", response.data)
        self.assertNotIn(b'name="start_date"', response.data)

    def test_autobio_shows_the_generate_form_once_a_key_exists(self):
        self.openai_path.parent.mkdir(parents=True)
        self.openai_path.write_text("sk-fake\n")

        response = self.client.get("/autobio")

        self.assertIn(b'name="start_date"', response.data)
        self.assertNotIn(b"/settings/connect", response.data)

    def test_entries_and_summaries_still_show_even_when_ungated_content_is_hidden(self):
        # Existing content is still viewable without a live key — only the
        # *generate* action needs one.
        with self.conn:
            repository.save_autobio_draft(
                self.conn, date="2026-08-14", segments=[], draft_text="Already written.", has_unlabeled=False
            )

        response = self.client.get("/autobio/daily")

        self.assertIn(b"Already written.", response.data)


if __name__ == "__main__":
    unittest.main()
