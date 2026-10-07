from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .layout import questions_by_part


COLORS = {
    "correct": (55, 181, 91),
    "incorrect": (58, 65, 220),
    "unanswered": (31, 184, 230),
    "invalid": (22, 125, 245),
}


def annotate_results(
    image: np.ndarray,
    layout: dict[str, Any],
    results: dict[str, dict[str, dict[str, Any]]],
) -> np.ndarray:
    annotated = image.copy()
    radius = int(layout["detection"].get("annotation_radius", 11))
    for part, questions in questions_by_part(layout).items():
        for question in questions:
            if question.question not in results.get(part, {}):
                continue
            item = results[part][question.question]
            status = item["status"]
            color = COLORS[status]
            selected = set(item.get("selected", []))
            if selected:
                targets = [bubble for bubble in question.bubbles if bubble.option in selected]
            else:
                targets = list(question.bubbles)
            for bubble in targets:
                cv2.circle(annotated, (bubble.x, bubble.y), radius, color, 2, cv2.LINE_AA)
    return annotated


def annotate_calibration(
    image: np.ndarray,
    layout: dict[str, Any],
    detections: dict[str, dict[str, Any]],
) -> np.ndarray:
    annotated = image.copy()
    for part, questions in questions_by_part(layout).items():
        for question in questions:
            selected = set(detections[part][question.question].selected)
            for bubble in question.bubbles:
                color = (0, 255, 0) if bubble.option in selected else (255, 200, 0)
                cv2.circle(annotated, (bubble.x, bubble.y), 10, color, 1, cv2.LINE_AA)
                cv2.putText(
                    annotated,
                    f"{question.question}{bubble.option}",
                    (bubble.x - 8, bubble.y - 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.22,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return annotated
