from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

import backend.main as main
from backend.storage import TestRepository


def synthetic_unknown_sheet() -> tuple[bytes, dict[str, dict[str, str]]]:
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    grids = [
        ("part1", [80, 110, 140, 170, 200], [120, 150, 180, 210]),
        ("part2", [350, 390, 430, 470, 510, 550, 590], [330, 365, 400]),
    ]
    key: dict[str, dict[str, str]] = {"part1": {}, "part2": {}}
    for part, xs, ys in grids:
        options = list("ABCDE") if part == "part1" else list("0123456")
        labels = (
            [str(index + 1) for index in range(len(ys))]
            if part == "part1"
            else [f"2.1{chr(ord('a') + index)}" for index in range(len(ys))]
        )
        for row, (label, y) in enumerate(zip(labels, ys, strict=True)):
            for x in xs:
                cv2.circle(image, (x, y), 10, (20, 70, 130), 2, cv2.LINE_AA)
            selected = row % len(xs)
            cv2.circle(image, (xs[selected], y), 8, (0, 0, 0), -1, cv2.LINE_AA)
            key[part][label] = options[selected]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes(), key


def test_unknown_bubble_sheet_uses_template_free_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "repository", TestRepository(tmp_path / "tests.json"))
    client = TestClient(main.app)
    created = client.post(
        "/api/tests",
        json={
            "name": "Unknown dynamic sheet",
            "part_counts": {"part1": 4, "part2": 3},
        },
    )
    assert created.status_code == 201
    test_id = created.json()["test"]["id"]
    image, answer_key = synthetic_unknown_sheet()
    assert (
        client.post(f"/api/tests/{test_id}/answer-key", json=answer_key).status_code
        == 200
    )

    response = client.post(
        f"/api/tests/{test_id}/grade",
        files={"image": ("unknown.png", image, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["alignment"]["template_id"] == "generic_dynamic"
    assert payload["alignment"]["method"] == "generic-bubble-grid"
    assert payload["student_answers"] == answer_key
    assert payload["summary"]["total"]["correct"] == 7
    assert len(payload["detected_answers"]["part1"]["1"]["options"]) == 5
    assert len(payload["detected_answers"]["part2"]["2.1a"]["options"]) == 7
