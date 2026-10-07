from pathlib import Path

import cv2
import numpy as np

from backend.grading import grade_detections
from backend.omr.processor import OMRProcessor


ROOT = Path(__file__).resolve().parents[2]
SCORING = {
    "correct_points": 1,
    "incorrect_points": 0,
    "unanswered_points": 0,
    "invalid_points": 0,
}


def test_supplied_sheet_answers_are_detected_without_answer_key() -> None:
    result = OMRProcessor().detect_bytes((ROOT / "samples" / "filled.png").read_bytes())
    answers = result.public_answers()

    assert len(answers["part1"]) == 60
    assert len(answers["part2"]) == 30
    assert answers["part1"]["1"] == "B"
    assert answers["part1"]["60"] == "C"
    assert answers["part2"]["2.1a"] == "8"
    assert answers["part2"]["2.4f"] == "7"
    assert result.aligned_image.shape[:2] == (1344, 1290)


def test_teacher_key_drives_grading() -> None:
    result = OMRProcessor().detect_bytes((ROOT / "samples" / "filled.png").read_bytes())
    teacher_key = {
        "part1": {"1": "B", "2": "B", "3": "D"},
        "part2": {"2.1a": "8", "2.1b": "6"},
    }
    graded = grade_detections(result.detections, teacher_key, SCORING)

    assert graded["grading"]["part1"]["1"]["status"] == "correct"
    assert graded["grading"]["part1"]["2"]["status"] == "incorrect"
    assert graded["summary"]["part1"]["correct"] == 2
    assert graded["summary"]["part2"]["correct"] == 1
    assert graded["summary"]["total"]["correct"] == 3
    assert graded["summary"]["total"]["questions"] == 5
    assert graded["summary"]["total"]["score"] == 60.0


def test_camera_framing_rotation_and_jpeg_compression_are_aligned() -> None:
    processor = OMRProcessor()
    raw = (ROOT / "samples" / "filled.png").read_bytes()
    reference = processor.detect_bytes(raw).public_answers()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    camera_transform = np.float32([[0.78, 0.035, 58], [-0.018, 0.75, 52]])
    photographed = cv2.warpAffine(
        image,
        camera_transform,
        (1290, 1344),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(235, 238, 235),
    )
    ok, encoded = cv2.imencode(".jpg", photographed, [cv2.IMWRITE_JPEG_QUALITY, 72])
    assert ok

    result = processor.detect_bytes(encoded.tobytes())

    assert result.alignment.method == "registration-lattice-affine"
    assert result.alignment.matched_markers >= 40
    assert result.public_answers() == reference


def test_real_low_quality_photo_detects_every_marked_row() -> None:
    result = OMRProcessor().detect_bytes((ROOT / "filled_lowquality_real.png").read_bytes())
    answers = result.public_answers()

    assert result.alignment.method == "registration-lattice-affine"
    assert result.alignment.matched_markers >= 45
    assert all(answer not in {"-", "Multiple"} for answer in answers["part1"].values())
    assert all(answer not in {"-", "Multiple"} for answer in answers["part2"].values())
    assert {
        "1": answers["part1"]["1"],
        "30": answers["part1"]["30"],
        "31": answers["part1"]["31"],
        "60": answers["part1"]["60"],
    } == {"1": "D", "30": "B", "31": "C", "60": "A"}
    assert {
        "2.1a": answers["part2"]["2.1a"],
        "2.2a": answers["part2"]["2.2a"],
        "2.3a": answers["part2"]["2.3a"],
        "2.4f": answers["part2"]["2.4f"],
    } == {"2.1a": "3", "2.2a": "1", "2.3a": "2", "2.4f": "3"}


def test_unanswered_and_multiple_are_classified() -> None:
    processor = OMRProcessor()
    image = cv2.imread(str(ROOT / "samples" / "filled.png"))
    cv2.circle(image, (263, 429), 8, (255, 255, 255), -1)
    cv2.circle(image, (241, 454), 8, (0, 0, 0), -1)
    cv2.circle(image, (284, 454), 8, (0, 0, 0), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = processor.detect_bytes(encoded.tobytes())
    graded = grade_detections(
        result.detections,
        {"part1": {"1": "B", "2": "A"}, "part2": {"2.1a": "8"}},
        SCORING,
    )

    assert graded["grading"]["part1"]["1"]["status"] == "unanswered"
    assert graded["grading"]["part1"]["2"]["status"] == "invalid"
