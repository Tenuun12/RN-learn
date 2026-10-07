from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as main
from backend.storage import TestRepository


ROOT = Path(__file__).resolve().parents[2]


def test_complete_teacher_workflow(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "repository", TestRepository(tmp_path / "tests.json"))
    client = TestClient(main.app)

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
    assert client.get(payload["annotated_image_url"]).status_code == 200
