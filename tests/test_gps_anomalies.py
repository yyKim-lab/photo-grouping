import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import gps_anomalies as ga  # noqa: E402

BASE_TIME = datetime(2026, 4, 12, 2, 15, 0)

# Real coordinates from the case that motivated this check: photos
# timestamped ~34s apart, one in 경기도 광주시, the rest in 전남 강진군.
GWANGJU_GYEONGGI = (37.34700, 127.28200)
GANGJIN_JEONNAM = (34.64200, 126.76700)


def _photo(photo_id, *, seconds=0, lat=37.5, lng=127.0, faces=(1,)):
    return ga.PhotoLocationSample(
        photo_id=photo_id,
        taken_at=BASE_TIME + timedelta(seconds=seconds),
        lat=lat,
        lng=lng,
        face_cluster_ids=frozenset(faces),
    )


class FindGpsAnomaliesTests(unittest.TestCase):
    def test_flags_the_real_world_stale_gps_case(self):
        photos = [
            _photo("a", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1]),
            _photo("b", seconds=34, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1]),
        ]

        anomalies = ga.find_gps_anomalies(photos)

        self.assertEqual(len(anomalies), 1)
        self.assertGreater(anomalies[0].implied_speed_kmh, 10_000)
        self.assertEqual(anomalies[0].shared_face_cluster_ids, frozenset({1}))

    def test_does_not_flag_plausible_movement(self):
        # ~1km apart, 10 minutes — an easy walk.
        photos = [
            _photo("a", seconds=0, lat=37.5000, lng=127.0000),
            _photo("b", seconds=600, lat=37.5090, lng=127.0000),
        ]

        self.assertEqual(ga.find_gps_anomalies(photos), [])

    def test_shared_face_grades_high_confidence(self):
        photos = [
            _photo("a", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1], faces=(1,)),
            _photo("b", seconds=34, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(1,)),
        ]

        anomalies = ga.find_gps_anomalies(photos)
        self.assertEqual(anomalies[0].confidence, ga.Confidence.HIGH)

    def test_no_shared_face_still_flagged_but_low_confidence(self):
        # Same impossible jump, different people. Still reported — real
        # data showed filtering these out loses genuine errors in photos
        # with nobody in them — but graded LOW, since these two photos
        # could legitimately be unrelated (a forwarded photo, a screenshot).
        photos = [
            _photo("a", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1], faces=(1,)),
            _photo("b", seconds=34, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(2,)),
        ]

        anomalies = ga.find_gps_anomalies(photos)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].confidence, ga.Confidence.LOW)

    def test_photo_with_no_faces_at_all_is_still_flagged(self):
        # The exact case that motivated grading over filtering: a
        # mislocated photo with nobody in it.
        photos = [
            _photo("has_people", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1], faces=(1,)),
            _photo("no_people", seconds=34, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=()),
        ]

        anomalies = ga.find_gps_anomalies(photos)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].confidence, ga.Confidence.LOW)

    def test_require_shared_face_restricts_to_high_confidence_only(self):
        photos = [
            _photo("a", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1], faces=(1,)),
            _photo("b", seconds=34, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(2,)),
        ]

        self.assertEqual(ga.find_gps_anomalies(photos, require_shared_face=True), [])

    def test_ignores_pairs_outside_the_time_window(self):
        photos = [
            _photo("a", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1]),
            # 3 hours later — plenty of time to actually make the trip.
            _photo("b", seconds=10_800, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1]),
        ]

        self.assertEqual(ga.find_gps_anomalies(photos), [])

    def test_gps_jitter_below_tolerance_is_never_flagged(self):
        # Identical timestamps a few metres apart: infinite implied speed,
        # but this is ordinary GPS noise, not a real anomaly.
        photos = [
            _photo("a", seconds=0, lat=37.50000, lng=127.00000),
            _photo("b", seconds=0, lat=37.50010, lng=127.00000),
        ]

        self.assertEqual(ga.find_gps_anomalies(photos), [])

    def test_same_instant_far_apart_is_flagged(self):
        photos = [
            _photo("a", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1]),
            _photo("b", seconds=0, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1]),
        ]

        anomalies = ga.find_gps_anomalies(photos)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].implied_speed_kmh, float("inf"))

    def test_photos_without_gps_or_timestamp_are_skipped(self):
        photos = [
            ga.PhotoLocationSample(photo_id="no_gps", taken_at=BASE_TIME, lat=None, lng=None),
            ga.PhotoLocationSample(photo_id="no_time", taken_at=None, lat=37.5, lng=127.0),
            _photo("ok", seconds=0),
        ]

        self.assertEqual(ga.find_gps_anomalies(photos), [])

    def test_input_order_does_not_matter(self):
        later = _photo("b", seconds=34, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1])
        earlier = _photo("a", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1])

        anomalies = ga.find_gps_anomalies([later, earlier])

        self.assertEqual(len(anomalies), 1)
        # Sorted internally, so the pair is always reported oldest-first.
        self.assertEqual(anomalies[0].photo_a_id, "a")
        self.assertGreater(anomalies[0].time_delta_seconds, 0)


class PhotosNeedingReviewTests(unittest.TestCase):
    def test_ranks_the_true_outlier_above_its_neighbours(self):
        # Mirrors the real case: several correctly-located photos, one
        # mislocated one that conflicts with all of them.
        good = [
            _photo(f"good_{i}", seconds=i * 5, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1])
            for i in range(4)
        ]
        bad = _photo("bad", seconds=10, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1])

        checklist = ga.photos_needing_review(ga.find_gps_anomalies(good + [bad]))

        self.assertEqual(checklist[0].photo_id, "bad")
        # The outlier conflicts with every good photo; each good photo
        # conflicts only with the outlier.
        self.assertEqual(checklist[0].conflict_count, len(good))
        for item in checklist[1:]:
            self.assertLess(item.conflict_count, checklist[0].conflict_count)

    def test_heavily_conflicted_low_photo_outranks_barely_conflicted_high(self):
        # Regression test for a ranking flaw real data exposed: a genuinely
        # mislocated photo with no faces in it (hence LOW) had 23 conflicts,
        # while merely-adjacent correctly-located photos had 5 each and
        # graded HIGH. Confidence-first ordering buried the real outlier
        # below 17 innocent photos, so conflict count has to lead.
        low_crowd = [
            _photo(f"low_{i}", seconds=i, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1], faces=(i + 100,))
            for i in range(5)
        ]
        low_outlier = _photo("low_outlier", seconds=2, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(999,))

        # A separate pair, far later in time so it can't interact with the
        # crowd above, that does share a face — HIGH but few conflicts.
        high_a = _photo("high_a", seconds=5_000, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1], faces=(7,))
        high_b = _photo("high_b", seconds=5_030, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(7,))

        checklist = ga.photos_needing_review(
            ga.find_gps_anomalies(low_crowd + [low_outlier, high_a, high_b])
        )

        self.assertEqual(checklist[0].photo_id, "low_outlier")
        self.assertEqual(checklist[0].confidence, ga.Confidence.LOW)
        self.assertGreater(checklist[0].conflict_count, checklist[1].conflict_count)

    def test_confidence_breaks_ties_at_equal_conflict_count(self):
        # Two photos, same conflict count, differing confidence: HIGH first.
        photos = [
            _photo("anchor", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1], faces=(1,)),
            _photo("high_peer", seconds=30, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(1,)),
            _photo("low_peer", seconds=31, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(2,)),
        ]

        checklist = ga.photos_needing_review(ga.find_gps_anomalies(photos))
        peers = [i for i in checklist if i.photo_id in ("high_peer", "low_peer")]

        self.assertEqual(peers[0].conflict_count, peers[1].conflict_count)
        self.assertEqual(peers[0].photo_id, "high_peer")

    def test_one_high_confidence_pair_promotes_a_photo(self):
        # A photo in both HIGH and LOW pairs inherits HIGH.
        photos = [
            _photo("shared", seconds=0, lat=GANGJIN_JEONNAM[0], lng=GANGJIN_JEONNAM[1], faces=(1,)),
            _photo("same_person", seconds=30, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(1,)),
            _photo("stranger", seconds=60, lat=GWANGJU_GYEONGGI[0], lng=GWANGJU_GYEONGGI[1], faces=(2,)),
        ]

        checklist = ga.photos_needing_review(ga.find_gps_anomalies(photos))

        shared = next(i for i in checklist if i.photo_id == "shared")
        self.assertEqual(shared.confidence, ga.Confidence.HIGH)

    def test_empty_input_gives_empty_checklist(self):
        self.assertEqual(ga.photos_needing_review([]), [])


if __name__ == "__main__":
    unittest.main()
