from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .errors import AlignmentConfidenceError, LayoutDetectionError
from .layout import iter_questions


@dataclass(frozen=True)
class AlignmentResult:
    image: np.ndarray
    matched_markers: int
    expected_markers: int
    confidence: float
    method: str


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Return four points in top-left, top-right, bottom-right, bottom-left order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[int(np.argmin(sums))]
    ordered[2] = points[int(np.argmax(sums))]
    ordered[1] = points[int(np.argmin(differences))]
    ordered[3] = points[int(np.argmax(differences))]
    return ordered


def _corner_marker_candidates(
    image: np.ndarray, config: dict[str, Any]
) -> list[tuple[float, float, float, float]]:
    """Find compact, solid black rectangles near the four page corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = int(config.get("dark_threshold", 90))
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    minimum_side = min(height, width) * float(
        config.get("minimum_marker_side_ratio", 0.012)
    )
    maximum_side = min(height, width) * float(
        config.get("maximum_marker_side_ratio", 0.075)
    )
    minimum_extent = float(config.get("minimum_marker_extent", 0.72))
    candidates: list[tuple[float, float, float, float]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        short_side = min(box_width, box_height)
        long_side = max(box_width, box_height)
        if short_side < minimum_side or long_side > maximum_side:
            continue
        area = float(cv2.contourArea(contour))
        extent = area / max(float(box_width * box_height), 1.0)
        aspect = box_width / max(float(box_height), 1.0)
        perimeter = float(cv2.arcLength(contour, True))
        approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1.0)
        if not (
            0.62 <= aspect <= 1.62
            and extent >= minimum_extent
            and solidity >= 0.88
            and len(approximation) == 4
            and cv2.isContourConvex(approximation)
        ):
            continue
        center, size, _ = cv2.minAreaRect(contour)
        marker_area = max(float(size[0] * size[1]), 1.0)
        candidates.append((float(center[0]), float(center[1]), marker_area, extent))
    return candidates


def _select_corner_markers(
    image: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray | None, int, float]:
    candidates = _corner_marker_candidates(image, config)
    height, width = image.shape[:2]
    zone = float(config.get("corner_zone_ratio", 0.28))
    corners = (
        (0.0, 0.0, lambda x, y: x <= width * zone and y <= height * zone),
        (float(width), 0.0, lambda x, y: x >= width * (1 - zone) and y <= height * zone),
        (
            float(width),
            float(height),
            lambda x, y: x >= width * (1 - zone) and y >= height * (1 - zone),
        ),
        (
            0.0,
            float(height),
            lambda x, y: x <= width * zone and y >= height * (1 - zone),
        ),
    )
    selected: list[tuple[float, float, float, float]] = []
    diagonal = max(float(np.hypot(width, height)), 1.0)
    for corner_x, corner_y, inside in corners:
        available = [candidate for candidate in candidates if inside(candidate[0], candidate[1])]
        if not available:
            continue
        selected.append(
            min(
                available,
                key=lambda item: (
                    np.hypot(item[0] - corner_x, item[1] - corner_y) / diagonal
                    + (1.0 - item[3]) * 0.25
                ),
            )
        )
    if len(selected) != 4:
        return None, len(selected), 0.0

    points = np.float32([(item[0], item[1]) for item in selected])
    areas = [item[2] for item in selected]
    extents = [item[3] for item in selected]
    quadrilateral_area = abs(float(cv2.contourArea(points)))
    image_area = float(width * height)
    area_ratio = quadrilateral_area / max(image_area, 1.0)
    size_ratio = max(areas) / max(min(areas), 1.0)
    minimum_area = float(config.get("minimum_marker_quad_area_ratio", 0.55))
    maximum_size_ratio = float(config.get("maximum_marker_size_ratio", 2.5))
    if (
        area_ratio < minimum_area
        or size_ratio > maximum_size_ratio
        or not cv2.isContourConvex(points)
    ):
        return None, 4, 0.0
    coverage_score = min(1.0, area_ratio / max(minimum_area, 0.001))
    size_score = min(1.0, maximum_size_ratio / max(size_ratio, 0.001))
    confidence = float(np.mean(extents)) * coverage_score * size_score
    return points, 4, confidence


def _document_boundary(
    image: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray | None, float]:
    """Locate a visible sheet boundary when one or more corner markers are obscured."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(gray.shape[0] * gray.shape[1])
    minimum_area_ratio = float(config.get("fallback_minimum_area_ratio", 0.35))
    best: tuple[float, np.ndarray, float] | None = None
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        area_ratio = area / max(image_area, 1.0)
        if area_ratio < minimum_area_ratio:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue
        points = _order_quad(approximation[:, 0, :])
        rectangularity = area / max(abs(float(cv2.contourArea(points))), 1.0)
        confidence = min(1.0, area_ratio / 0.75) * min(1.0, rectangularity)
        score = area_ratio * confidence
        if best is None or score > best[0]:
            best = (score, points, confidence)
    if best is None:
        return None, 0.0
    return best[1], best[2]


def _corner_homography(
    image: np.ndarray,
    marker_image: np.ndarray,
    layout: dict[str, Any],
    config: dict[str, Any],
    validate_bubble_grid: bool = True,
) -> AlignmentResult:
    width = int(layout["image_width"])
    height = int(layout["image_height"])
    source, marker_count, confidence = _select_corner_markers(marker_image, config)
    method = "corner-marker-homography"
    if source is not None:
        target = np.float32(config["marker_targets"])
        if target.shape != (4, 2):
            raise ValueError("corner-marker alignment requires four marker_targets")
    else:
        boundary, boundary_confidence = _document_boundary(marker_image, config)
        if boundary is None:
            message = (
                f"Only {marker_count} of 4 corner markers were detected, and the document "
                "boundary could not be recovered reliably. Retake the photo with all four "
                "black corner markers and the full page visible."
                if marker_count
                else "The four black corner markers and document boundary could not be detected reliably."
            )
            error_type = AlignmentConfidenceError if marker_count else LayoutDetectionError
            raise error_type(message)
        source = boundary
        target = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        confidence = boundary_confidence
        method = "document-boundary-homography"

    minimum_confidence = float(config.get("minimum_alignment_confidence", 0.65))
    if confidence < minimum_confidence:
        raise AlignmentConfidenceError(
            "The answer sheet was found, but alignment confidence is too low. "
            "Retake the photo with the full page and corner markers visible."
        )
    matrix = cv2.getPerspectiveTransform(np.float32(source), target)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-9:
        raise AlignmentConfidenceError("The detected answer sheet perspective is degenerate.")
    aligned = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    recognition = config.get("recognition", {})
    if validate_bubble_grid and recognition.get("enabled", False):
        matched_bubbles, expected_bubbles = _circle_grid_match(aligned, layout, config)
        bubble_confidence = matched_bubbles / max(expected_bubbles, 1)
        if bubble_confidence < float(recognition.get("minimum_match_ratio", 0.72)):
            raise AlignmentConfidenceError(
                "The corner markers were found, but the answer regions did not align "
                "with the expected sheet. Check that the full answer sheet is visible."
            )
        confidence = min(confidence, bubble_confidence)
    return AlignmentResult(
        aligned,
        marker_count,
        4,
        min(1.0, confidence),
        method,
    )


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
    anchor: tuple[float, float] | None = None,
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
            if anchor is not None:
                anchor_offset = np.float32(anchor) - array[first]
                anchor_distance = abs(
                    float(vector[0] * anchor_offset[1] - vector[1] * anchor_offset[0])
                ) / length
                if anchor_distance > tolerance * 1.5:
                    continue
            offsets = array - array[first]
            distances = np.abs(vector[0] * offsets[:, 1] - vector[1] * offsets[:, 0]) / length
            indices = np.flatnonzero(distances <= tolerance)
            if len(indices) < 3:
                continue
            if anchor is not None:
                anchor_gaps = np.linalg.norm(array[indices] - np.float32(anchor), axis=1)
                if float(anchor_gaps.min()) > max(tolerance * 3, 12):
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
    left = config.get("left_marker_grid")
    expected_top = [tuple(map(float, point)) for point in config.get("top_markers", [])]
    if not left or len(rail) < 6 or len(expected_top) < 4:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    # The first top marker is also the first marker of the vertical rail. Anchoring
    # the horizontal search at that shared corner prevents long diagonal rows in
    # page content from winning when a QR footer changes the input aspect ratio.
    top = _dedupe_sorted(
        _best_line_inliers(candidates, "horizontal", tolerance, anchor=rail[0]),
        axis=0,
    )
    if len(top) < 4:
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


def _circle_grid_match(
    image: np.ndarray, layout: dict[str, Any], config: dict[str, Any]
) -> tuple[int, int]:
    recognition = config.get("recognition", {})
    expected = np.float32(
        [
            (bubble.x, bubble.y)
            for question in iter_questions(layout)
            for bubble in question.bubbles
        ]
    )
    if len(expected) == 0:
        return 0, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(recognition.get("hough_dp", 1)),
        minDist=float(recognition.get("hough_min_distance", 12)),
        param1=float(recognition.get("hough_param1", 100)),
        param2=float(recognition.get("hough_param2", 18)),
        minRadius=int(recognition.get("minimum_radius", 7)),
        maxRadius=int(recognition.get("maximum_radius", 13)),
    )
    if circles is None:
        return 0, len(expected)
    detected = np.float32(circles[0, :, :2])
    maximum_distance = float(recognition.get("maximum_center_distance", 4))
    matched = sum(
        float(np.min(np.linalg.norm(detected - point, axis=1))) <= maximum_distance
        for point in expected
    )
    return matched, len(expected)


def align_image(
    image: np.ndarray,
    layout: dict[str, Any],
    registration_image: np.ndarray | None = None,
    *,
    validate_bubble_grid: bool = True,
) -> AlignmentResult:
    width = int(layout["image_width"])
    height = int(layout["image_height"])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    marker_source = registration_image if registration_image is not None else image
    config = layout.get("alignment", {})
    if config.get("method") == "corner-marker-homography":
        return _corner_homography(
            image,
            marker_source,
            layout,
            config,
            validate_bubble_grid=validate_bubble_grid,
        )
    marker_image = cv2.resize(marker_source, (width, height), interpolation=cv2.INTER_AREA)
    if not config.get("enabled", True):
        recognition = config.get("recognition", {})
        if recognition.get("enabled", False):
            matched, expected_count = _circle_grid_match(resized, layout, config)
            confidence = matched / max(expected_count, 1)
            if confidence < float(recognition.get("minimum_match_ratio", 0.75)):
                raise LayoutDetectionError(
                    "This image does not match the selected answer-sheet template."
                )
            return AlignmentResult(
                resized,
                matched,
                expected_count,
                confidence,
                "template-circle-grid",
            )
        return AlignmentResult(resized, 0, 0, 1.0, "resize")

    expected = _expected_markers(config)
    candidates = _find_registration_markers(marker_image, config)
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
    singular_values = np.linalg.svd(matrix[:, :2], compute_uv=False)
    smallest = float(singular_values.min())
    largest = float(singular_values.max())
    if smallest < 0.35 or largest > 3.0 or largest / max(smallest, 0.001) > 3.0:
        raise LayoutDetectionError(
            "The detected answer sheet alignment is distorted. Please check the image framing."
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
