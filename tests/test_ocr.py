"""OCR candidate filtering — no model load, tests the pure filtering/encoding
logic against synthetic RapidOCR-shaped output."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import ocr  # noqa: E402


def _fake_result(pairs):
    """pairs: [(text, score), ...] -> object shaped like RapidOCR's result."""
    result = MagicMock()
    result.txts = [t for t, _ in pairs]
    result.scores = [s for _, s in pairs]
    return result


class ReadTextCandidatesTests(unittest.TestCase):
    @patch("photo_grouping.ocr._get_engine")
    def test_ranks_by_confidence_descending(self, mock_engine):
        mock_engine.return_value = MagicMock(
            return_value=_fake_result([("abc", 0.5), ("xyz", 0.9), ("def", 0.7)])
        )

        candidates = ocr.read_text_candidates(Path("fake.jpg"))

        self.assertEqual([c["text"] for c in candidates], ["xyz", "def", "abc"])

    @patch("photo_grouping.ocr._get_engine")
    def test_filters_below_confidence_floor(self, mock_engine):
        mock_engine.return_value = MagicMock(
            return_value=_fake_result([("real", 0.9), ("noise", 0.2)])
        )

        candidates = ocr.read_text_candidates(Path("fake.jpg"))

        self.assertEqual([c["text"] for c in candidates], ["real"])

    @patch("photo_grouping.ocr._get_engine")
    def test_filters_mostly_numeric_text(self, mock_engine):
        # A building's construction year, a phone number, opening hours —
        # all real OCR output, none of them a place name.
        mock_engine.return_value = MagicMock(
            return_value=_fake_result([("1949", 0.99), ("055-010", 0.9), ("코롬방", 0.78)])
        )

        candidates = ocr.read_text_candidates(Path("fake.jpg"))

        self.assertEqual([c["text"] for c in candidates], ["코롬방"])

    @patch("photo_grouping.ocr._get_engine")
    def test_filters_out_of_length_range(self, mock_engine):
        mock_engine.return_value = MagicMock(
            return_value=_fake_result(
                [
                    ("a", 0.9),  # too short
                    ("x" * 30, 0.9),  # too long
                    ("코롬방", 0.9),  # just right
                ]
            )
        )

        candidates = ocr.read_text_candidates(Path("fake.jpg"))

        self.assertEqual([c["text"] for c in candidates], ["코롬방"])

    @patch("photo_grouping.ocr._get_engine")
    def test_deduplicates_repeated_text(self, mock_engine):
        mock_engine.return_value = MagicMock(
            return_value=_fake_result([("코롬방", 0.9), ("코롬방", 0.8), ("다른곳", 0.7)])
        )

        candidates = ocr.read_text_candidates(Path("fake.jpg"))

        self.assertEqual([c["text"] for c in candidates], ["코롬방", "다른곳"])

    @patch("photo_grouping.ocr._get_engine")
    def test_respects_the_limit(self, mock_engine):
        mock_engine.return_value = MagicMock(
            return_value=_fake_result([(f"word{i}", 0.9 - i * 0.01) for i in range(20)])
        )

        candidates = ocr.read_text_candidates(Path("fake.jpg"), limit=3)

        self.assertEqual(len(candidates), 3)

    @patch("photo_grouping.ocr._get_engine")
    def test_no_text_detected_returns_empty(self, mock_engine):
        mock_engine.return_value = MagicMock(return_value=_fake_result([]))

        self.assertEqual(ocr.read_text_candidates(Path("fake.jpg")), [])

    @patch("photo_grouping.ocr._get_engine")
    def test_real_storefront_case_surfaces_the_name_within_default_limit(self, mock_engine):
        # Regression test for the exact case that motivated raising the
        # default limit from 5 to 8: on a real photo, five higher-confidence
        # non-name strings (a fire hydrant label, a street sign fragment, an
        # automatic-door notice, a slogan, a torn "OPEN" sign) outranked the
        # actual bakery name "코롬방", which placed 6th.
        mock_engine.return_value = MagicMock(
            return_value=_fake_result(
                [
                    ("소화전", 0.99),
                    ("XIAMEN-RO", 0.98),
                    ("자동문", 0.94),
                    ("더:인피닛요트체험", 0.92),
                    ("DPEN-", 0.89),
                    ("코롬방", 0.78),
                ]
            )
        )

        candidates = ocr.read_text_candidates(Path("fake.jpg"))

        self.assertIn("코롬방", [c["text"] for c in candidates])


class EncodeDecodeTests(unittest.TestCase):
    def test_roundtrips_through_json(self):
        candidates = [{"text": "코롬방", "confidence": 0.78}]
        self.assertEqual(ocr.decode_candidates(ocr.encode_candidates(candidates)), candidates)

    def test_empty_list_encodes_to_none(self):
        self.assertIsNone(ocr.encode_candidates([]))

    def test_decode_none_is_empty_list(self):
        self.assertEqual(ocr.decode_candidates(None), [])

    def test_decode_tolerates_a_plain_string_from_an_older_version(self):
        self.assertEqual(ocr.decode_candidates("코롬방"), [{"text": "코롬방", "confidence": None}])

    def test_korean_survives_the_roundtrip_unescaped(self):
        # ensure_ascii=False: a stored value should be human-readable if
        # someone opens the database directly, not \uXXXX escapes.
        encoded = ocr.encode_candidates([{"text": "코롬방제과점", "confidence": 0.5}])
        self.assertIn("코롬방제과점", encoded)


if __name__ == "__main__":
    unittest.main()
