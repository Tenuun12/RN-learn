from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.omr.errors import AlignmentConfidenceError
from backend.omr.layout import questions_by_part
from backend.omr.processor import AutoOMRProcessor, OMRProcessor
from backend.storage import TestRepository


ROOT = Path(__file__).resolve().parents[2]
LAYOUT_PATH = ROOT / "backend" / "config" / "answer_sheet_40_28_layout.json"


def _normalized_sheet(
    *,
    obscure_top_left: bool = False,
    layout_path: Path = LAYOUT_PATH,
) -> tuple[np.ndarray, dict[str, dict[str, str]]]:
    processor = OMRProcessor(layout_path)
    layout = processor.layout
    image = np.full(
        (layout["image_height"], layout["image_width"], 3), 255, dtype=np.uint8
    )
    for x, y in layout["alignment"]["marker_targets"]:
        cv2.rectangle(image, (x - 14, y - 14), (x + 14, y + 14), (0, 0, 0), -1)

    key: dict[str, dict[str, str]] = {"part1": {}, "part2": {}}
    for part, questions in questions_by_part(layout).items():
        for row_index, question in enumerate(questions):
            for bubble in question.bubbles:
                cv2.circle(image, (bubble.x, bubble.y), 12, (15, 80, 190), 2, cv2.LINE_AA)
            selected = question.bubbles[row_index % len(question.bubbles)]
            cv2.circle(image, (selected.x, selected.y), 9, (0, 0, 0), -1, cv2.LINE_AA)
            key[part][question.question] = selected.option

    if obscure_top_left:
        x, y = layout["alignment"]["marker_targets"][0]
        cv2.rectangle(image, (x - 22, y - 22), (x + 22, y + 22), (255, 255, 255), -1)
    return image, key


def _photograph(sheet: np.ndarray, *, shadow: bool = False) -> np.ndarray:
    if shadow:
        lighting = np.linspace(0.68, 1.0, sheet.shape[1], dtype=np.float32)[None, :, None]
        sheet = np.clip(sheet.astype(np.float32) * lighting, 0, 255).astype(np.uint8)
    source = np.float32(
        [[0, 0], [sheet.shape[1] - 1, 0], [sheet.shape[1] - 1, sheet.shape[0] - 1], [0, sheet.shape[0] - 1]]
    )
    destination = np.float32([[125, 145], [1340, 70], [1390, 1830], [55, 1890]])
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        sheet,
        matrix,
        (1450, 1950),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(165, 165, 165),
    )


def test_new_tests_default_to_40_and_32_questions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "repository", TestRepository(tmp_path / "tests.json"))
    client = TestClient(main.app)

    layout = client.get("/api/layout").json()
    assert layout["default_part_counts"] == {"part1": 40, "part2": 32}
    created = client.post("/api/tests", json={"name": "Automatic counts"})

    assert created.status_code == 201
    assert created.json()["test"]["part_counts"] == {"part1": 40, "part2": 32}
    assert len(created.json()["test"]["parts"]["part1"]["question_labels"]) == 40
    assert len(created.json()["test"]["parts"]["part2"]["question_labels"]) == 32


def test_straight_sheet_uses_the_same_normalized_coordinates() -> None:
    processor = OMRProcessor(LAYOUT_PATH)
    sheet, expected = _normalized_sheet()

    result = processor.detect_image(sheet)

    assert result.alignment.method == "corner-marker-homography"
    assert result.aligned_image.shape[:2] == sheet.shape[:2]
    assert result.public_answers() == expected


def test_corner_markers_correct_rotation_skew_perspective_and_shadow() -> None:
    processor = OMRProcessor(LAYOUT_PATH)
    sheet, expected = _normalized_sheet()
    photographed = _photograph(sheet, shadow=True)

    result = processor.detect_image(photographed)

    assert result.alignment.method == "corner-marker-homography"
    assert result.alignment.matched_markers == 4
    assert result.alignment.confidence >= 0.65
    assert result.aligned_image.shape[:2] == (1725, 1249)
    assert result.public_answers() == expected


def test_document_boundary_fallback_handles_an_obscured_marker() -> None:
    processor = OMRProcessor(LAYOUT_PATH)
    sheet, expected = _normalized_sheet(obscure_top_left=True)
    photographed = _photograph(sheet)

    result = processor.detect_image(photographed)

    assert result.alignment.method == "document-boundary-homography"
    assert result.alignment.confidence >= 0.65
    assert result.public_answers() == expected


def test_partial_markers_without_a_boundary_return_a_clear_error() -> None:
    processor = OMRProcessor(LAYOUT_PATH)
    sheet, _ = _normalized_sheet(obscure_top_left=True)

    with pytest.raises(AlignmentConfidenceError, match="3 of 4 corner markers"):
        processor.detect_image(sheet)


def test_automatic_reader_detects_all_rows_on_latest_sheet_after_perspective() -> None:
    latest_layout = ROOT / "backend" / "config" / "answer_sheet_40_32_layout.json"
    sheet, expected = _normalized_sheet(layout_path=latest_layout)
    photographed = _photograph(sheet)

    result = AutoOMRProcessor().detect_image(photographed)

    assert len(result.read.detections["part1"]) == 40
    assert len(result.read.detections["part2"]) == 32
    assert result.read.public_answers() == expected


def test_unknown_question_counts_use_rectified_dynamic_grid() -> None:
    latest_layout = ROOT / "backend" / "config" / "answer_sheet_40_32_layout.json"
    sheet, _ = _normalized_sheet(layout_path=latest_layout)
    # A distinct pair of marks changes no printed geometry, while the extra
    # row below forces the saved layouts to fall through to grid discovery.
    extra_y = 514 + 20 * 40
    cv2.circle(sheet, (158, extra_y), 12, (20, 70, 130), 2, cv2.LINE_AA)
    for x in [194, 230, 262, 298]:
        cv2.circle(sheet, (x, extra_y), 12, (20, 70, 130), 2, cv2.LINE_AA)
    cv2.circle(sheet, (158, extra_y), 9, (0, 0, 0), -1, cv2.LINE_AA)

    result = AutoOMRProcessor().detect_image(sheet)

    assert result.template_id == "generic_dynamic"
    assert len(result.read.detections["part1"]) >= 41
    assert result.read.public_answers()["part1"]["41"] == "A"
