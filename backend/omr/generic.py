from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from string import ascii_uppercase
from typing import Any

import cv2
import numpy as np

from .alignment import AlignmentResult
from .errors import LayoutDetectionError


@dataclass(frozen=True)
class BubbleCandidate:
    x: float
    y: float
    diameter: float


@dataclass(frozen=True)
class BubbleGrid:
    x_centers: tuple[int, ...]
    y_centers: tuple[int, ...]
    matched: int
    expected: int


@dataclass(frozen=True)
class GenericLayoutResult:
    image: np.ndarray
    layout: dict[str, Any]
    alignment: AlignmentResult


def _deduplicate(candidates: list[BubbleCandidate]) -> list[BubbleCandidate]:
    kept: list[BubbleCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.diameter, reverse=True):
        duplicate = any(
            np.hypot(candidate.x - other.x, candidate.y - other.y)
            < min(candidate.diameter, other.diameter) * 0.45
            for other in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def _bubble_candidates(image: np.ndarray) -> tuple[list[BubbleCandidate], float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_side = min(gray.shape)
    candidates: list[BubbleCandidate] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        diameter = float(max(width, height))
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
        aspect = width / max(height, 1)
        if (
            minimum_side * 0.008 <= diameter <= minimum_side * 0.06
            and 0.65 <= aspect <= 1.45
            and area >= 15
            and circularity >= 0.42
        ):
            candidates.append(
                BubbleCandidate(x + width / 2, y + height / 2, diameter)
            )

    candidates = _deduplicate(candidates)
    if not candidates:
        raise LayoutDetectionError("No repeated answer bubbles were found in this image.")
    diameter_bins = Counter(round(item.diameter / 2) * 2 for item in candidates)
    dominant_diameter, support = diameter_bins.most_common(1)[0]
    if support < 12:
        raise LayoutDetectionError("No repeated answer-bubble pattern was found in this image.")
    filtered = [
        item
        for item in candidates
        if dominant_diameter * 0.75 <= item.diameter <= dominant_diameter * 1.30
    ]
    return filtered, float(dominant_diameter)


def _cluster_axis(
    candidates: list[BubbleCandidate], axis: str, tolerance: float
) -> list[tuple[float, list[BubbleCandidate]]]:
    coordinate = (lambda item: item.x) if axis == "x" else (lambda item: item.y)
    clusters: list[list[BubbleCandidate]] = []
    for candidate in sorted(candidates, key=coordinate):
        if clusters:
            center = float(np.median([coordinate(item) for item in clusters[-1]]))
            if abs(coordinate(candidate) - center) <= tolerance:
                clusters[-1].append(candidate)
                continue
        clusters.append([candidate])
    return [
        (float(np.median([coordinate(item) for item in cluster])), cluster)
        for cluster in clusters
    ]


def _regular_column_runs(
    candidates: list[BubbleCandidate], diameter: float, image_width: int
) -> list[tuple[float, ...]]:
    tolerance = diameter * 0.35
    columns = [
        (center, cluster)
        for center, cluster in _cluster_axis(candidates, "x", tolerance)
        if len(cluster) >= 3
    ]
    centers = [center for center, _ in columns]
    support = {center: len(cluster) for center, cluster in columns}
    proposed: list[tuple[float, int, tuple[float, ...]]] = []
    for first in range(len(centers)):
        for second in range(first + 1, min(first + 5, len(centers))):
            spacing = centers[second] - centers[first]
            if spacing < diameter * 0.9 or spacing > image_width * 0.12:
                continue
            run = [centers[first]]
            expected = centers[first] + spacing
            while expected <= centers[-1] + tolerance:
                nearby = [center for center in centers if abs(center - expected) <= tolerance]
                if not nearby:
                    break
                chosen = min(nearby, key=lambda center: abs(center - expected))
                if chosen in run:
                    break
                run.append(chosen)
                expected += spacing
            if 3 <= len(run) <= 20:
                total_support = sum(support[center] for center in run)
                proposed.append((len(run) * total_support, total_support, tuple(run)))

    selected: list[tuple[float, ...]] = []
    for _, _, run in sorted(proposed, reverse=True):
        rounded = {round(center) for center in run}
        overlaps = any(
            len(rounded & {round(center) for center in existing})
            >= min(len(run), len(existing)) * 0.5
            for existing in selected
        )
        if not overlaps:
            selected.append(run)
    return selected


def _rows_for_run(
    candidates: list[BubbleCandidate], run: tuple[float, ...], diameter: float
) -> BubbleGrid | None:
    tolerance = diameter * 0.35
    nearby = [
        candidate
        for candidate in candidates
        if min(abs(candidate.x - x) for x in run) <= tolerance * 1.5
    ]
    rows: list[tuple[int, int]] = []
    for center, cluster in _cluster_axis(nearby, "y", tolerance):
        hits = sum(
            any(abs(candidate.x - x) <= tolerance * 1.5 for candidate in cluster)
            for x in run
        )
        if hits >= max(3, len(run) * 0.6):
            rows.append((round(center), hits))
    if len(rows) < 2:
        return None
    return BubbleGrid(
        tuple(round(center) for center in run),
        tuple(center for center, _ in rows),
        sum(hits for _, hits in rows),
        len(run) * len(rows),
    )


def _option_labels(count: int) -> list[str]:
    if count <= 5:
        return list(ascii_uppercase[:count])
    numeric = [str(index) for index in range(min(count, 10))]
    extras = ["-", "."]
    while len(numeric) < count:
        index = len(numeric) - 10
        numeric.append(extras[index] if index < len(extras) else str(len(numeric)))
    return numeric


def _question_labels(
    part: str, count: int, provided: dict[str, list[str]] | None
) -> list[str]:
    requested = list((provided or {}).get(part, []))
    if part == "part1":
        defaults = [str(index + 1) for index in range(count)]
    else:
        defaults = [
            f"2.{index // 8 + 1}{ascii_uppercase[index % 8].lower()}"
            for index in range(count)
        ]
    labels = requested[:count]
    used = set(labels)
    for default in defaults[len(labels) :]:
        candidate = default
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"detected_{suffix}"
        labels.append(candidate)
        used.add(candidate)
    return labels


def detect_generic_layout(
    image: np.ndarray,
    registration_image: np.ndarray | None = None,
    question_labels: dict[str, list[str]] | None = None,
) -> GenericLayoutResult:
    """Discover repeated bubble grids without a saved page layout."""
    source = registration_image if registration_image is not None else image
    candidates, diameter = _bubble_candidates(source)
    runs = _regular_column_runs(candidates, diameter, image.shape[1])
    grids = [
        grid
        for run in runs
        if (grid := _rows_for_run(candidates, run, diameter)) is not None
    ]
    if not grids:
        raise LayoutDetectionError(
            "This page does not contain a readable repeated bubble grid."
        )

    # Short alphabetic runs are conventional multiple-choice rows; longer runs
    # are treated as numeric/symbol choices. Geometry, not a template id, decides.
    by_part: dict[str, list[BubbleGrid]] = {"part1": [], "part2": []}
    for grid in grids:
        by_part["part1" if len(grid.x_centers) <= 5 else "part2"].append(grid)
    by_part["part1"].sort(key=lambda grid: (grid.x_centers[0], grid.y_centers[0]))
    by_part["part2"].sort(key=lambda grid: (grid.y_centers[0], grid.x_centers[0]))

    parts: dict[str, Any] = {}
    total_matched = 0
    total_expected = 0
    for part, part_grids in by_part.items():
        if not part_grids:
            parts[part] = {"label": part.title(), "options": [], "grids": []}
            continue
        option_count = Counter(len(grid.x_centers) for grid in part_grids).most_common(1)[0][0]
        part_grids = [grid for grid in part_grids if len(grid.x_centers) == option_count]
        total_rows = sum(len(grid.y_centers) for grid in part_grids)
        labels = _question_labels(part, total_rows, question_labels)
        offset = 0
        serialized_grids = []
        for grid in part_grids:
            count = len(grid.y_centers)
            for row_index, y in enumerate(grid.y_centers):
                serialized_grids.append(
                    {
                        "question_labels": [labels[offset + row_index]],
                        "count": 1,
                        "x_centers": list(grid.x_centers),
                        "y_start": y,
                        "y_spacing": 1,
                    }
                )
            offset += count
            total_matched += grid.matched
            total_expected += grid.expected
        parts[part] = {
            "label": "Part 1" if part == "part1" else "Part 2",
            "options": _option_labels(option_count),
            "grids": serialized_grids,
        }

    question_count = sum(len(part["grids"]) for part in parts.values())
    if question_count < 3 or total_expected == 0:
        raise LayoutDetectionError("Too few repeated answer rows were found on this page.")
    confidence = total_matched / total_expected
    if confidence < 0.60:
        raise LayoutDetectionError("The detected bubble grid is too incomplete to read safely.")

    sample_radius = max(3, round(diameter * 0.32))
    layout = {
        "id": "generic_dynamic",
        "name": "Automatically discovered bubble sheet",
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "alignment": {"enabled": False, "required": False},
        "detection": {
            "sample_radius": sample_radius,
            "center_search_radius": max(1, round(diameter * 0.08)),
            "illumination_kernel": 41,
            "annotation_radius": max(sample_radius + 3, round(diameter * 0.60)),
            "dark_pixel_threshold": 120,
            "adaptive_dark_threshold": {"enabled": False},
            "filled_threshold": 0.45,
            "weak_filled_threshold": 0.25,
            "minimum_score_margin": 0.12,
            "empty_threshold": 0.18,
        },
        "parts": parts,
    }
    alignment = AlignmentResult(
        image,
        total_matched,
        total_expected,
        confidence,
        "generic-bubble-grid",
    )
    return GenericLayoutResult(image, layout, alignment)
