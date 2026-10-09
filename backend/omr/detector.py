from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .layout import Question, questions_by_part


@dataclass(frozen=True)
class Detection:
    answer: str | None
    selected: tuple[str, ...]
    fill_scores: dict[str, float]
    confidence: float
    state: str


def _registration_points(layout: dict[str, Any]) -> list[tuple[float, float]]:
    alignment = layout.get("alignment", {})
    points = [tuple(map(float, point)) for point in alignment.get("top_markers", [])]
    left = alignment.get("left_marker_grid")
    if left:
        points.extend(
            (
                float(left["x"]),
                float(left["y_start"] + index * left["y_spacing"]),
            )
            for index in range(int(left["count"]))
        )
    return points


def _adaptive_dark_threshold(
    normalized_gray: np.ndarray,
    layout: dict[str, Any],
    detection: dict[str, Any],
) -> int:
    adaptive = detection.get("adaptive_dark_threshold", {})
    if not adaptive.get("enabled", True):
        return int(detection["dark_pixel_threshold"])
    half_width = int(adaptive.get("marker_half_width", 6))
    half_height = int(adaptive.get("marker_half_height", 3))
    values: list[np.ndarray] = []
    height, width = normalized_gray.shape
    for x_value, y_value in _registration_points(layout):
        x, y = int(round(x_value)), int(round(y_value))
        if x - half_width < 0 or y - half_height < 0 or x + half_width >= width or y + half_height >= height:
            continue
        values.append(
            normalized_gray[
                y - half_height : y + half_height + 1,
                x - half_width : x + half_width + 1,
            ].ravel()
        )
    if not values:
        return int(detection["dark_pixel_threshold"])
    marker_pixels = np.concatenate(values)
    marker_edge = float(
        np.percentile(marker_pixels, float(adaptive.get("marker_percentile", 95)))
    )
    threshold = round(marker_edge + float(adaptive.get("margin", 40)))
    return int(
        np.clip(
            threshold,
            int(adaptive.get("minimum", 70)),
            int(adaptive.get("maximum", 155)),
        )
    )


def _circle_fill_ratio(
    gray: np.ndarray,
    x: int,
    y: int,
    radius: int,
    dark_pixel_threshold: int,
) -> float:
    height, width = gray.shape
    if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
        return 0.0
    roi = gray[y - radius : y + radius + 1, x - radius : x + radius + 1]
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    circle = xx * xx + yy * yy <= radius * radius
    return float(np.count_nonzero(roi[circle] < dark_pixel_threshold) / np.count_nonzero(circle))


def _best_circle_fill_ratio(
    gray: np.ndarray,
    x: int,
    y: int,
    radius: int,
    dark_pixel_threshold: int,
    search_radius: int,
) -> float:
    return max(
        _circle_fill_ratio(gray, x + dx, y + dy, radius, dark_pixel_threshold)
        for dy in range(-search_radius, search_radius + 1)
        for dx in range(-search_radius, search_radius + 1)
    )


def detect_question(gray: np.ndarray, question: Question, detection: dict[str, Any]) -> Detection:
    radius = int(detection["sample_radius"])
    dark_threshold = int(detection["dark_pixel_threshold"])
    filled_threshold = float(detection["filled_threshold"])
    weak_threshold = float(detection.get("weak_filled_threshold", filled_threshold))
    minimum_margin = float(detection.get("minimum_score_margin", 0.12))
    empty_threshold = float(detection["empty_threshold"])
    search_radius = int(detection.get("center_search_radius", 0))
    scores = {
        bubble.option: round(
            _best_circle_fill_ratio(
                gray,
                bubble.x,
                bubble.y,
                radius,
                dark_threshold,
                search_radius,
            ),
            4,
        )
        for bubble in question.bubbles
    }
    selected = tuple(option for option, score in scores.items() if score >= filled_threshold)
    if len(selected) == 1:
        answer: str | None = selected[0]
        confidence = scores[answer]
        state = "selected"
    elif len(selected) > 1:
        answer = "Multiple"
        confidence = min(scores[option] for option in selected)
        state = "multiple"
    else:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_option, peak = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        weak_selected = tuple(option for option, score in scores.items() if score >= weak_threshold)
        if peak >= weak_threshold and peak - second >= minimum_margin:
            answer = top_option
            selected = (top_option,)
            confidence = max(0.5, min(peak / max(filled_threshold, 0.001), 1.0))
            state = "selected"
        elif len(weak_selected) > 1:
            answer = "Multiple"
            selected = weak_selected
            confidence = min(scores[option] for option in weak_selected)
            state = "multiple"
        else:
            answer = None
            if peak <= empty_threshold:
                confidence = 1.0 - peak
                state = "blank"
            else:
                confidence = max(
                    0.0,
                    (weak_threshold - peak) / max(weak_threshold - empty_threshold, 0.001),
                )
                state = "uncertain"
    return Detection(answer, selected, scores, round(confidence, 4), state)


def detect_answers(image: np.ndarray, layout: dict[str, Any]) -> dict[str, dict[str, Detection]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detection = layout["detection"]
    kernel_size = int(detection.get("illumination_kernel", 0))
    if kernel_size >= 3:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.divide(gray, np.maximum(background, 1), scale=255)
    # Light denoising preserves filled disks while reducing camera/JPEG speckles.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    runtime_detection = {
        **detection,
        "dark_pixel_threshold": _adaptive_dark_threshold(gray, layout, detection),
    }
    result: dict[str, dict[str, Detection]] = {}
    for part, questions in questions_by_part(layout).items():
        result[part] = {
            question.question: detect_question(gray, question, runtime_detection)
            for question in questions
        }
    return result
