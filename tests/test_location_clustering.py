import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import location_clustering as lc  # noqa: E402


class HaversineDistanceTests(unittest.TestCase):
    def test_same_point_is_zero(self):
        self.assertAlmostEqual(lc.haversine_distance_m(37.5, 127.0, 37.5, 127.0), 0.0, places=3)

    def test_known_distance_seoul_to_busan_roughly_325km(self):
        # Seoul city hall ~37.5665,126.9780 ; Busan city hall ~35.1796,129.0756
        distance = lc.haversine_distance_m(37.5665, 126.9780, 35.1796, 129.0756)
        self.assertGreater(distance, 300_000)
        self.assertLess(distance, 350_000)

    def test_small_offset_within_100m(self):
        # ~0.0009 degrees latitude is roughly 100m
        distance = lc.haversine_distance_m(37.5000, 127.0000, 37.5009, 127.0000)
        self.assertGreater(distance, 90)
        self.assertLess(distance, 110)


class AssignOrCreateTests(unittest.TestCase):
    def test_first_point_creates_a_new_cluster(self):
        clusters: list[lc.LocationClusterCandidate] = []
        cluster = lc.assign_or_create(37.5, 127.0, clusters, photo_index=0)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(cluster.centroid_lat, 37.5)
        self.assertEqual(cluster.centroid_lng, 127.0)
        self.assertEqual(cluster.photo_indices, [0])

    def test_nearby_point_joins_existing_cluster(self):
        clusters: list[lc.LocationClusterCandidate] = []
        lc.assign_or_create(37.5000, 127.0000, clusters, photo_index=0)
        # ~50m away — well within the default 150m threshold
        cluster = lc.assign_or_create(37.50045, 127.0000, clusters, photo_index=1, threshold_m=150)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(cluster.photo_indices, [0, 1])
        self.assertEqual(cluster.count, 2)

    def test_far_point_creates_a_second_cluster(self):
        clusters: list[lc.LocationClusterCandidate] = []
        lc.assign_or_create(37.5665, 126.9780, clusters, photo_index=0)  # Seoul
        lc.assign_or_create(35.1796, 129.0756, clusters, photo_index=1)  # Busan

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].photo_indices, [0])
        self.assertEqual(clusters[1].photo_indices, [1])

    def test_centroid_moves_toward_new_points_as_a_running_mean(self):
        clusters: list[lc.LocationClusterCandidate] = []
        lc.assign_or_create(37.5000, 127.0000, clusters, photo_index=0)
        cluster = lc.assign_or_create(37.5000, 127.0010, clusters, photo_index=1, threshold_m=150)

        # Running mean of two points should land halfway between them.
        self.assertAlmostEqual(cluster.centroid_lng, 127.0005, places=6)

    def test_joins_nearest_of_several_clusters_not_just_the_first(self):
        clusters: list[lc.LocationClusterCandidate] = []
        lc.assign_or_create(37.0000, 127.0000, clusters, photo_index=0)  # far cluster A
        lc.assign_or_create(38.0000, 127.0000, clusters, photo_index=1)  # far cluster B
        # A tiny nudge from cluster B, still far from A
        cluster = lc.assign_or_create(38.0001, 127.0000, clusters, photo_index=2, threshold_m=150)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(cluster.photo_indices, [1, 2])


if __name__ == "__main__":
    unittest.main()
