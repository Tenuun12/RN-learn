import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import backend.main as main
from backend.storage import TestRepository


ROOT = Path(__file__).resolve().parents[2]


def _sheet_with_qr(data: str) -> bytes:
    sheet = cv2.imread(str(ROOT / "samples" / "filled.png"))
    qr = cv2.QRCodeEncoder_create().encode(data)
    qr = cv2.resize(qr, (330, 330), interpolation=cv2.INTER_NEAREST)
    height, width = sheet.shape[:2]
    canvas = np.full((height, width + 410, 3), 255, dtype=np.uint8)
    canvas[:, :width] = sheet
    canvas[80:410, width + 40 : width + 370] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    return encoded.tobytes()


def test_complete_teacher_workflow(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "repository", TestRepository(tmp_path / "tests.json"))
    client = TestClient(main.app)

    api_index = client.get("/api")
    assert api_index.status_code == 200
    assert api_index.json()["documentation"]["swagger"] == "/api/docs"
    assert client.get("/api/docs").status_code == 200
    openapi = client.get("/api/openapi.json")
    assert openapi.status_code == 200
    assert "/api/tests/{test_id}/grade" in openapi.json()["paths"]

    created = client.post(
        "/api/tests",
        json={"name": "Manual grading test", "part_counts": {"part1": 3, "part2": 2}},
    )
    assert created.status_code == 201
    test_id = created.json()["test"]["id"]

    missing = client.post(
        f"/api/tests/{test_id}/answer-key",
        json={"part1": {"1": "B"}, "part2": {"2.1a": "8"}},
    )
    assert missing.status_code == 422

    answer_key = {
        "part1": {"1": "B", "2": "B", "3": "D"},
        "part2": {"2.1a": "8", "2.1b": "6"},
    }
    saved = client.post(f"/api/tests/{test_id}/answer-key", json=answer_key)
    assert saved.status_code == 200
    assert client.get(f"/api/tests/{test_id}/answer-key").json()["answer_key"] == answer_key

    edited_key = {**answer_key, "part1": {**answer_key["part1"], "2": "A"}}
    assert client.post(f"/api/tests/{test_id}/answer-key", json=edited_key).status_code == 200
    assert client.get(f"/api/tests/{test_id}/answer-key").json()["answer_key"] == edited_key
    assert client.post(f"/api/tests/{test_id}/answer-key", json=answer_key).status_code == 200

    with (ROOT / "samples" / "filled.png").open("rb") as handle:
        graded = client.post(
            f"/api/tests/{test_id}/grade",
            files={"image": ("filled.png", handle, "image/png")},
        )
    assert graded.status_code == 200
    payload = graded.json()
    assert payload["answer_key"] == answer_key
    assert payload["student_answers"]["part1"] == {"1": "B", "2": "A", "3": "D"}
    assert payload["summary"]["total"]["questions"] == 5
    assert payload["summary"]["total"]["correct"] == 3
    assert payload["summary"]["total"]["score"] == 60.0
    assert payload["qr_detected"] is False
    assert payload["qr_data"] is None
    assert payload["qr_url"] is None
    assert payload["qr_codes"] == []
    assert payload["qr_image_data_url"] is None
    assert payload["qr_image_data_urls"] == []
    assert payload["annotated_image_data_url"].startswith("data:image/jpeg;base64,")
    assert client.get(payload["annotated_image_url"]).status_code == 200

    with (ROOT / "samples" / "filled.png").open("rb") as handle:
        stateless = client.post(
            "/api/tests/restored-after-cold-start/grade",
            data={
                "answer_key_json": json.dumps(answer_key),
                "test_config_json": json.dumps(
                    {
                        "name": "Locally restored key",
                        "part_counts": {"part1": 3, "part2": 2},
                    }
                ),
            },
            files={"image": ("filled.png", handle, "image/png")},
        )
    assert stateless.status_code == 200
    assert stateless.json()["answer_key"] == answer_key
    assert stateless.json()["summary"]["total"]["score"] == 60.0

    expected_url = "https://example.com/grade/student-42?exam=midterm"
    with_qr = client.post(
        f"/api/tests/{test_id}/grade",
        files={"image": ("filled-with-qr.png", _sheet_with_qr(expected_url), "image/png")},
    )
    assert with_qr.status_code == 200
    qr_payload = with_qr.json()
    assert qr_payload["student_answers"]["part1"] == payload["student_answers"]["part1"]
    assert qr_payload["student_answers"]["part2"] == payload["student_answers"]["part2"]
    assert qr_payload["qr_detected"] is True
    assert qr_payload["qr_data"] == expected_url
    assert qr_payload["qr_url"] == expected_url
    assert qr_payload["qr_codes"] == [{"data": expected_url, "url": expected_url}]
    assert qr_payload["qr_image_data_url"].startswith("data:image/png;base64,")
    assert len(qr_payload["qr_image_data_urls"]) == 1

    def broken_qr_decoder(_image):
        raise cv2.error("Simulated QR decoder failure")

    monkeypatch.setattr(main, "detect_qr_codes", broken_qr_decoder)
    with (ROOT / "samples" / "filled.png").open("rb") as handle:
        qr_failure = client.post(
            f"/api/tests/{test_id}/grade",
            files={"image": ("filled.png", handle, "image/png")},
        )
    assert qr_failure.status_code == 200
    assert qr_failure.json()["qr_detected"] is False
    assert qr_failure.json()["qr_image_data_url"] is None
    assert qr_failure.json()["student_answers"] == payload["student_answers"]
