import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import face_clustering as fc  # noqa: E402

DIMS = 512


def _unit(*, seed: float) -> list[float]:
    """A deterministic unit-length vector, standing in for an ArcFace
    embedding (which arrives L2-normalized)."""
    values = [math.sin(seed + i * 0.7) for i in range(DIMS)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def _at_similarity(base: list[float], target_similarity: float) -> list[float]:
    """A unit vector whose cosine similarity to `base` is target_similarity,
    built by mixing `base` with an orthogonal direction."""
    other = _unit(seed=99.0)
    # Gram-Schmidt: strip the component of `other` along `base`.
    dot = sum(a * b for a, b in zip(base, other))
    perp = [o - dot * b for o, b in zip(other, base)]
    perp_norm = math.sqrt(sum(v * v for v in perp))
    perp = [v / perp_norm for v in perp]

    scale = math.sqrt(max(0.0, 1 - target_similarity**2))
    return [target_similarity * b + scale * p for b, p in zip(base, perp)]


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors_are_one(self):
        v = _unit(seed=1.0)
        self.assertAlmostEqual(fc.cosine_similarity(v, v), 1.0, places=6)

    def test_constructed_similarity_is_accurate(self):
        base = _unit(seed=2.0)
        probe = _at_similarity(base, 0.4)
        self.assertAlmostEqual(fc.cosine_similarity(base, probe), 0.4, places=6)

    def test_normalizes_both_sides(self):
        # A cluster centroid is a mean of unit vectors and is NOT unit
        # length; without normalizing, similarity would sag purely as a
        # function of how varied a cluster's members are.
        base = _unit(seed=3.0)
        scaled = [v * 0.25 for v in base]
        self.assertAlmostEqual(fc.cosine_similarity(base, scaled), 1.0, places=6)

    def test_zero_vector_does_not_divide_by_zero(self):
        self.assertEqual(fc.cosine_similarity(_unit(seed=4.0), [0.0] * DIMS), 0.0)


class AssignTests(unittest.TestCase):
    def test_no_clusters_yet_returns_none(self):
        self.assertIsNone(fc.assign(_unit(seed=1.0), []))

    def test_joins_a_cluster_above_the_threshold(self):
        base = _unit(seed=1.0)
        cluster = fc.FaceClusterCentroid(cluster_id=1, centroid=base)
        similar = _at_similarity(base, fc.DEFAULT_SIMILARITY_THRESHOLD + 0.1)

        self.assertIs(fc.assign(similar, [cluster]), cluster)

    def test_rejects_a_face_below_the_threshold(self):
        base = _unit(seed=1.0)
        cluster = fc.FaceClusterCentroid(cluster_id=1, centroid=base)
        dissimilar = _at_similarity(base, fc.DEFAULT_SIMILARITY_THRESHOLD - 0.1)

        self.assertIsNone(fc.assign(dissimilar, [cluster]))

    def test_picks_the_most_similar_of_several_clusters(self):
        base = _unit(seed=1.0)
        close = fc.FaceClusterCentroid(cluster_id=1, centroid=base)
        far = fc.FaceClusterCentroid(cluster_id=2, centroid=_at_similarity(base, 0.0))
        probe = _at_similarity(base, 0.9)

        self.assertIs(fc.assign(probe, [far, close]), close)

    def test_assign_does_not_mutate_clusters(self):
        base = _unit(seed=1.0)
        cluster = fc.FaceClusterCentroid(cluster_id=1, centroid=base)
        before = list(cluster.centroid)

        fc.assign(_at_similarity(base, 0.95), [cluster])

        # Creating/updating rows is the caller's job — assign only decides.
        self.assertEqual(cluster.centroid, before)
        self.assertEqual(cluster.count, 1)


class CentroidTests(unittest.TestCase):
    def test_add_moves_the_centroid_as_a_running_mean(self):
        a, b = _unit(seed=1.0), _unit(seed=5.0)
        cluster = fc.FaceClusterCentroid(cluster_id=1, centroid=a)

        cluster.add(b)

        self.assertEqual(cluster.count, 2)
        for i in range(DIMS):
            self.assertAlmostEqual(cluster.centroid[i], (a[i] + b[i]) / 2, places=9)

    def test_running_mean_weights_by_existing_membership(self):
        a, b = _unit(seed=1.0), _unit(seed=5.0)
        cluster = fc.FaceClusterCentroid(cluster_id=1, centroid=a, count=3)

        cluster.add(b)

        # A new member shifts a large cluster less than a small one.
        for i in range(DIMS):
            self.assertAlmostEqual(cluster.centroid[i], (a[i] * 3 + b[i]) / 4, places=9)

    def test_a_drifted_centroid_still_compares_sensibly(self):
        # Regression guard for the normalization in cosine_similarity: after
        # averaging two dissimilar members the centroid is well short of
        # unit length, and an un-normalized dot product would understate
        # similarity to its own members.
        a = _unit(seed=1.0)
        b = _at_similarity(a, 0.5)
        cluster = fc.FaceClusterCentroid(cluster_id=1, centroid=a)
        cluster.add(b)

        self.assertGreater(fc.cosine_similarity(a, cluster.centroid), 0.8)


if __name__ == "__main__":
    unittest.main()
