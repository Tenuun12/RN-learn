from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import backend.main as main
from backend.omr.layout import questions_by_part
from backend.omr.processor import OMRProcessor
from backend.storage import TestRepository


ROOT = Path(__file__).resolve().parents[2]
LAYOUT_PATH = ROOT / "backend" / "config" / "school21_layout.json"


def synthetic_school21_sheet() -> tuple[bytes, dict[str, dict[str, str]]]:
    processor = OMRProcessor(LAYOUT_PATH)
    image = np.full(
        (processor.layout["image_height"], processor.layout["image_width"], 3),
        255,
        dtype=np.uint8,
    )
    key: dict[str, dict[str, str]] = {"part1": {}, "part2": {}}
    for part, questions in questions_by_part(processor.layout).items():
        for question in questions:
            for bubble in question.bubbles:
                cv2.circle(image, (bubble.x, bubble.y), 10, (0, 0, 0), 2, cv2.LINE_AA)
            selected = question.bubbles[0]
            cv2.circle(image, (selected.x, selected.y), 8, (0, 0, 0), -1, cv2.LINE_AA)
            key[part][question.question] = selected.option
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return encoded.tobytes(), key


def test_school21_template_is_detected_automatically_and_grades_all_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "repository", TestRepository(tmp_path / "tests.json"))
    client = TestClient(main.app)

    layout = client.get("/api/layout").json()
    template = next(
        item for item in layout["templates"] if item["id"] == "school21_70_32_v1"
    )
    assert template["parts"]["part1"]["maximum_questions"] == 70
    assert template["parts"]["part2"]["maximum_questions"] == 32

    created = client.post(
        "/api/tests",
        json={
            "name": "School 21 test",
            "part_counts": {"part1": 70, "part2": 32},
        },
    )
    assert created.status_code == 201
    test_id = created.json()["test"]["id"]
    assert created.json()["test"]["layout_id"] == "auto"

    image, answer_key = synthetic_school21_sheet()
    saved = client.post(f"/api/tests/{test_id}/answer-key", json=answer_key)
    assert saved.status_code == 200
    graded = client.post(
        f"/api/tests/{test_id}/grade",
        files={"image": ("school21.jpg", image, "image/jpeg")},
    )

    assert graded.status_code == 200
    payload = graded.json()
    assert payload["alignment"]["template_id"] == "school21_70_32_v1"
    assert payload["alignment"]["selection"] == "automatic"
    assert payload["alignment"]["method"] == "template-circle-grid"
    assert len(payload["detected_answers"]["part1"]) == 70
    assert len(payload["detected_answers"]["part2"]) == 32
    assert [
        option["label"]
        for option in payload["detected_answers"]["part1"]["1"]["options"]
    ] == ["A", "B", "C", "D", "E"]
    assert [
        option["label"]
        for option in payload["detected_answers"]["part2"]["2.1a"]["options"]
    ] == list("0123456789")
    assert payload["summary"]["total"]["questions"] == 102
    assert payload["summary"]["total"]["correct"] == 102
    assert payload["summary"]["total"]["score"] == 100.0


def test_school21_template_rejects_the_legacy_sheet() -> None:
    processor = OMRProcessor(LAYOUT_PATH)
    legacy = (ROOT / "samples" / "filled.png").read_bytes()

    try:
        processor.detect_bytes(legacy)
    except Exception as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("The legacy sheet must not match the School 21 template.")
