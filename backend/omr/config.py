from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_layout(path: str | Path | None = None) -> dict[str, Any]:
    return load_json(path or CONFIG_DIR / "omr_layout.json")
