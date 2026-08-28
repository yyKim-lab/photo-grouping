"""§4.5's spatial face-to-label matching, mocked at the detection/OCR
boundary so this tests the matching geometry, not the models themselves."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import seed_import  # noqa: E402

IMAGE_WIDTH, IMAGE_HEIGHT = 1000, 1000


def _face(x, y, w, h):
    return ({"x": x, "y": y, "width": w, "height": h}, [0.1] * 512)


def _text(text, left, top, right, bottom, confidence=0.9):
    return {"text": text, "confidence": confidence, "box": (left, top, right, bottom)}


class DetectSeedCandidatesTests(unittest.TestCase):
    def setUp(self):
        self._image_patch = patch("photo_grouping.seed_import.Image")
        mock_image_cls = self._image_patch.start()
        mock_image_cls.open.return_value.size = (IMAGE_WIDTH, IMAGE_HEIGHT)

    def tearDown(self):
        self._image_patch.stop()

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_matches_label_directly_below_a_face(self, mock_detect, mock_ocr):
        # Face at 100-200px square; label sitting just under it.
        mock_detect.return_value = [_face(0.1, 0.1, 0.1, 0.1)]
        mock_ocr.return_value = [_text("엄마", 100, 205, 200, 230)]

        candidates = seed_import.detect_seed_candidates(b"fake")

        self.assertEqual(candidates[0]["guessed_name"], "엄마")

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_ignores_text_above_the_face(self, mock_detect, mock_ocr):
        mock_detect.return_value = [_face(0.1, 0.3, 0.1, 0.1)]  # face at y=300-400
        mock_ocr.return_value = [_text("위쪽텍스트", 100, 50, 200, 80)]  # well above

        candidates = seed_import.detect_seed_candidates(b"fake")

        self.assertEqual(candidates[0]["guessed_name"], "")

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_ignores_text_too_far_below(self, mock_detect, mock_ocr):
        mock_detect.return_value = [_face(0.1, 0.1, 0.1, 0.1)]  # bottom at y=200, width=100
        mock_ocr.return_value = [_text("다음행캡션", 100, 600, 200, 630)]  # far below (next row)

        candidates = seed_import.detect_seed_candidates(b"fake")

        self.assertEqual(candidates[0]["guessed_name"], "")

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_ignores_text_horizontally_far_away(self, mock_detect, mock_ocr):
        mock_detect.return_value = [_face(0.1, 0.1, 0.1, 0.1)]  # x: 100-200
        mock_ocr.return_value = [_text("다른칸", 800, 205, 900, 230)]  # a different grid column

        candidates = seed_import.detect_seed_candidates(b"fake")

        self.assertEqual(candidates[0]["guessed_name"], "")

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_picks_the_closest_label_when_several_are_below(self, mock_detect, mock_ocr):
        mock_detect.return_value = [_face(0.1, 0.1, 0.1, 0.1)]
        mock_ocr.return_value = [
            _text("가까운이름", 100, 205, 200, 230),
            _text("먼캡션", 100, 350, 200, 380),
        ]

        candidates = seed_import.detect_seed_candidates(b"fake")

        self.assertEqual(candidates[0]["guessed_name"], "가까운이름")

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_multiple_faces_each_get_their_own_nearest_label(self, mock_detect, mock_ocr):
        # A row of two grid cells, each with its own label below it.
        mock_detect.return_value = [
            _face(0.0, 0.0, 0.1, 0.1),   # x: 0-100
            _face(0.5, 0.0, 0.1, 0.1),   # x: 500-600
        ]
        mock_ocr.return_value = [
            _text("첫번째", 0, 105, 100, 130),
            _text("두번째", 500, 105, 600, 130),
        ]

        candidates = seed_import.detect_seed_candidates(b"fake")

        names = {c["guessed_name"] for c in candidates}
        self.assertEqual(names, {"첫번째", "두번째"})

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_no_ocr_text_at_all_still_returns_the_face(self, mock_detect, mock_ocr):
        mock_detect.return_value = [_face(0.1, 0.1, 0.1, 0.1)]
        mock_ocr.return_value = []

        candidates = seed_import.detect_seed_candidates(b"fake")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["guessed_name"], "")

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_no_faces_detected_returns_empty(self, mock_detect, mock_ocr):
        mock_detect.return_value = []
        mock_ocr.return_value = [_text("텍스트", 0, 0, 100, 30)]

        self.assertEqual(seed_import.detect_seed_candidates(b"fake"), [])

    @patch("photo_grouping.seed_import.ocr.read_text_with_positions")
    @patch("photo_grouping.seed_import.face_embeddings.detect_faces_in_bytes")
    def test_embedding_is_carried_through_unchanged(self, mock_detect, mock_ocr):
        embedding = [0.42] * 512
        mock_detect.return_value = [({"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}, embedding)]
        mock_ocr.return_value = []

        candidates = seed_import.detect_seed_candidates(b"fake")

        self.assertEqual(candidates[0]["embedding"], embedding)


if __name__ == "__main__":
    unittest.main()
