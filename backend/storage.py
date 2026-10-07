from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "tests.json"


class TestNotFoundError(KeyError):
    pass


class AnswerKeyNotFoundError(KeyError):
    pass


class TestRepository:
    """Small JSON repository with an API that can later be replaced by a database."""

    __test__ = False

    def __init__(self, path: str | Path = DEFAULT_DATA_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"version": 1, "tests": {}, "active_test_id": None})

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, self.path)

    def list_tests(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            tests = list(data["tests"].values())
            return deepcopy(sorted(tests, key=lambda item: item["created_at"], reverse=True))

    def create_test(
        self,
        name: str,
        part_counts: dict[str, int],
        scoring: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        test_id = uuid4().hex[:12]
        record = {
            "id": test_id,
            "name": name.strip(),
            "part_counts": deepcopy(part_counts),
            "answer_key": None,
            "scoring": deepcopy(scoring),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            data = self._read()
            data["tests"][test_id] = record
            data["active_test_id"] = test_id
            self._write(data)
        return deepcopy(record)

    def get_test(self, test_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._read()["tests"].get(test_id)
            if record is None:
                raise TestNotFoundError(test_id)
            return deepcopy(record)

    def save_answer_key(self, test_id: str, answer_key: dict[str, dict[str, str]]) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            record = data["tests"].get(test_id)
            if record is None:
                raise TestNotFoundError(test_id)
            record["answer_key"] = deepcopy(answer_key)
            record["updated_at"] = datetime.now(UTC).isoformat()
            data["active_test_id"] = test_id
            self._write(data)
            return deepcopy(record)

    def get_answer_key(self, test_id: str) -> dict[str, dict[str, str]]:
        record = self.get_test(test_id)
        if record["answer_key"] is None:
            raise AnswerKeyNotFoundError(test_id)
        return deepcopy(record["answer_key"])
