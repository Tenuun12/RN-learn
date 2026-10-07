from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Bubble:
    part: str
    question: str
    option: str
    x: int
    y: int


@dataclass(frozen=True)
class Question:
    part: str
    question: str
    bubbles: tuple[Bubble, ...]


def _rounded_grid(start: float, spacing: float, count: int) -> list[int]:
    return [round(start + index * spacing) for index in range(count)]


def iter_questions(layout: dict[str, Any]) -> Iterator[Question]:
    for part_name, part in layout["parts"].items():
        options = part["options"]
        for grid in part["grids"]:
            ys = _rounded_grid(grid["y_start"], grid["y_spacing"], grid["count"])
            labels = grid.get("question_labels")
            if labels is None:
                start = grid["question_start"]
                labels = [str(start + index) for index in range(grid["count"])]

            for label, y in zip(labels, ys, strict=True):
                bubbles = tuple(
                    Bubble(part_name, label, option, int(x), y)
                    for option, x in zip(options, grid["x_centers"], strict=True)
                )
                yield Question(part_name, label, bubbles)


def questions_by_part(layout: dict[str, Any]) -> dict[str, list[Question]]:
    result = {part: [] for part in layout["parts"]}
    for question in iter_questions(layout):
        result[question.part].append(question)
    return result

