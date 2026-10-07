from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .errors import LayoutDetectionError


@dataclass(frozen=True)
class AlignmentResult:
    image: np.ndarray
    matched_markers: int
    expected_markers: int
    confidence: float
    method: str


def _expected_markers(config: dict[str, Any]) -> list[tuple[float, float]]:
    markers = [tuple(point) for point in config.get("top_markers", [])]
    left = config.get("left_marker_grid")
    if left:
        markers.extend(
            (float(left["x"]), float(left["y_start"] + index * left["y_spacing"]))
            for index in range(left["count"])
        )
    return markers


def _find_registration_markers(image: np.ndarray, config: dict[str, Any]) -> list[tuple[float, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, int(config.get("dark_threshold", 60)), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    candidates: list[tuple[float, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        extent = area / max(w * h, 1)
        aspect = w / max(h, 1)
        near_registration_edge = x < width * 0.32 or y < height * 0.16
        if (
            near_registration_edge
            and 6 <= w <= 42
            and 4 <= h <= 30
            and 28 <= area <= 900
            and extent >= 0.52
            and 1.18 <= aspect <= 3.8
        ):
            candidates.append((x + w / 2, y + h / 2))
    return candidates


def _best_line_inliers(
    points: list[tuple[float, float]],
    orientation: str,
    tolerance: float,
) -> list[tuple[float, float]]:
    """Find the strongest near-horizontal/vertical marker line without assuming its position."""
    if len(points) < 2:
        return []
    array = np.float32(points)
    best_indices: np.ndarray | None = None
    best_score = (-1, -1.0)
    for first in range(len(array) - 1):
        for second in range(first + 1, len(array)):
            vector = array[second] - array[first]
            dx, dy = float(vector[0]), float(vector[1])
            length = float(np.linalg.norm(vector))
            if length < 10:
                continue
            if orientation == "vertical" and abs(dy) < abs(dx) * 2.2:
                continue
            if orientation == "horizontal" and abs(dx) < abs(dy) * 2.2:
                continue
            offsets = array - array[first]
            distances = np.abs(vector[0] * offsets[:, 1] - vector[1] * offsets[:, 0]) / length
            indices = np.flatnonzero(distances <= tolerance)
            if len(indices) < 3:
                continue
            unit = vector / length
            projections = offsets[indices] @ unit
            span = float(np.ptp(projections))
            score = (len(indices), span)
            if score > best_score:
                best_score = score
                best_indices = indices
    if best_indices is None:
        return []
    return [points[int(index)] for index in best_indices]


def _dedupe_sorted(points: list[tuple[float, float]], axis: int) -> list[tuple[float, float]]:
    ordered = sorted(points, key=lambda point: point[axis])
    result: list[tuple[float, float]] = []
    for point in ordered:
        if not result or float(np.linalg.norm(np.subtract(point, result[-1]))) >= 4:
            result.append(point)
    return result


def _lattice_correspondences(
    candidates: list[tuple[float, float]],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Match the printed L-shaped registration lattice by order and spacing."""
    tolerance = float(config.get("line_tolerance", 5.0))
    rail = _dedupe_sorted(_best_line_inliers(candidates, "vertical", tolerance), axis=1)
    top = _dedupe_sorted(_best_line_inliers(candidates, "horizontal", tolerance), axis=0)
    left = config.get("left_marker_grid")
    expected_top = [tuple(map(float, point)) for point in config.get("top_markers", [])]
    if not left or len(rail) < 6 or len(top) < 4 or len(expected_top) < 4:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    source: list[tuple[float, float]] = []
    target: list[tuple[float, float]] = []

    rail_array = np.float32(rail)
    direction = rail_array[-1] - rail_array[0]
    direction /= max(float(np.linalg.norm(direction)), 0.001)
    positions = (rail_array - rail_array[0]) @ direction
    gaps = np.diff(positions)
    positive_gaps = gaps[gaps > 4]
    if len(positive_gaps) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    base_gap = float(np.percentile(positive_gaps, 35))
    used_indices: set[int] = set()
    for point, position in zip(rail, positions, strict=True):
        marker_index = int(round(float(position) / max(base_gap, 0.001)))
        if marker_index in used_indices or not 0 <= marker_index < int(left["count"]):
            continue
        used_indices.add(marker_index)
        source.append(point)
        target.append(
            (
                float(left["x"]),
                float(left["y_start"] + marker_index * left["y_spacing"]),
            )
        )

    # Top marker spacing is intentionally irregular, making its normalized pattern
    # a reliable identity signal even when the photographed page has large margins.
    top_array = np.float32(top)
    top_direction = top_array[-1] - top_array[0]
    top_direction /= max(float(np.linalg.norm(top_direction)), 0.001)
    source_positions = (top_array - top_array[0]) @ top_direction
    source_norm = source_positions / max(float(source_positions[-1]), 0.001)
    expected_array = np.float32(expected_top)
    expected_positions = expected_array[:, 0] - expected_array[0, 0]
    expected_norm = expected_positions / max(float(expected_positions[-1]), 0.001)
    used_source: set[int] = set()
    for expected_index, normalized in enumerate(expected_norm):
        available = [
            (abs(float(candidate_normalized - normalized)), source_index)
            for source_index, candidate_normalized in enumerate(source_norm)
            if source_index not in used_source
        ]
        if not available:
            break
        _, source_index = min(available)
        used_source.add(source_index)
        source.append(top[source_index])
        target.append(expected_top[expected_index])

    return np.float32(source), np.float32(target)


def _match_points(
    expected: list[tuple[float, float]],
    candidates: list[tuple[float, float]],
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    matches: list[tuple[float, int, int]] = []
    for expected_index, point in enumerate(expected):
        for candidate_index, candidate in enumerate(candidates):
            distance = float(np.linalg.norm(np.subtract(point, candidate)))
            if distance <= max_distance:
                matches.append((distance, expected_index, candidate_index))

    used_expected: set[int] = set()
    used_candidates: set[int] = set()
    source: list[tuple[float, float]] = []
    target: list[tuple[float, float]] = []
    for _, expected_index, candidate_index in sorted(matches):
        if expected_index in used_expected or candidate_index in used_candidates:
            continue
        used_expected.add(expected_index)
        used_candidates.add(candidate_index)
        source.append(candidates[candidate_index])
        target.append(expected[expected_index])
    return np.float32(source), np.float32(target)


def align_image(image: np.ndarray, layout: dict[str, Any]) -> AlignmentResult:
    width = int(layout["image_width"])
    height = int(layout["image_height"])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    config = layout.get("alignment", {})
    if not config.get("enabled", True):
        return AlignmentResult(resized, 0, 0, 1.0, "resize")

    expected = _expected_markers(config)
    candidates = _find_registration_markers(resized, config)
    source, target = _lattice_correspondences(candidates, config)
    method = "registration-lattice-affine"
    if len(source) < int(config.get("minimum_matches", 5)):
        source, target = _match_points(
            expected, candidates, float(config.get("max_match_distance", 48))
        )
        method = "registration-nearest-affine"
    match_count = len(source)
    minimum = int(config.get("minimum_matches", 5))
    confidence = match_count / max(len(expected), 1)

    if match_count < minimum:
        if config.get("required", True):
            raise LayoutDetectionError(
                "Unable to detect the answer sheet layout. Please upload a clearer image."
            )
        return AlignmentResult(resized, match_count, len(expected), confidence, "resize-fallback")

    # A full affine transform handles independent X/Y scale and shear introduced by
    # camera framing. It is stable for this L-shaped marker geometry, unlike a
    # projective homography whose four-point solution is underconstrained here.
    matrix, inliers = cv2.estimateAffine2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.get("ransac_threshold", 3.5)),
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None:
        raise LayoutDetectionError(
            "Unable to align the answer sheet. Please upload a clearer image."
        )
    aligned = cv2.warpAffine(
        resized,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    inlier_count = int(inliers.sum()) if inliers is not None else match_count
    return AlignmentResult(
        aligned,
        inlier_count,
        len(expected),
        inlier_count / max(len(expected), 1),
        method,
    )
