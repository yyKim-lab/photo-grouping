"""Seed-face matching (§3 step 5, §4.5): a match during ingestion must only
ever produce a suggestion, never an auto-assigned name — seed embeddings
come from low-res screenshot crops and are explicitly less reliable.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import face_clustering as fc  # noqa: E402

DIMS = 512


def _unit(seed: int) -> list[float]:
    import math

    values = [math.sin(seed + i * 0.7) for i in range(DIMS)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


class FindSeedMatchTests(unittest.TestCase):
    def test_returns_the_matching_name_above_threshold(self):
        target = _unit(1)
        seeds = [(1, "엄마", target)]

        self.assertEqual(fc.find_seed_match(target, seeds), "엄마")

    def test_returns_none_below_threshold(self):
        target = _unit(1)
        far = _unit(99)
        seeds = [(1, "엄마", far)]

        self.assertIsNone(fc.find_seed_match(target, seeds))

    def test_threshold_is_stricter_than_ordinary_cluster_assignment(self):
        # The spec calls seed embeddings less reliable than full-photo ones
        # — a false seed match mislabels a stranger with a real name, worse
        # than ordinary fragmentation, so the bar must be higher.
        self.assertGreater(fc.DEFAULT_SEED_MATCH_THRESHOLD, fc.DEFAULT_SIMILARITY_THRESHOLD)

    def test_returns_none_with_no_seed_faces(self):
        self.assertIsNone(fc.find_seed_match(_unit(1), []))

    def test_picks_the_best_match_among_several(self):
        target = _unit(1)
        seeds = [
            (1, "약한매치", _unit(50)),
            (2, "정답", target),
            (3, "다른사람", _unit(80)),
        ]

        self.assertEqual(fc.find_seed_match(target, seeds), "정답")

    def test_custom_threshold_is_respected(self):
        target = _unit(1)
        near = _unit(2)  # close but not identical
        seeds = [(1, "이름", near)]

        # With a very loose threshold it matches...
        self.assertEqual(fc.find_seed_match(target, seeds, threshold=-1.0), "이름")
        # ...with a very strict one it doesn't.
        self.assertIsNone(fc.find_seed_match(target, seeds, threshold=0.9999))


if __name__ == "__main__":
    unittest.main()
