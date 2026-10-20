from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alignment import AlignmentResult, align_image
from .config import discover_layout_paths, load_layout
from .detector import Detection, detect_answers
from .errors import AlignmentConfidenceError, InvalidImageError, LayoutDetectionError
from .generic import detect_generic_layout
from .layout import iter_questions


@dataclass(frozen=True)
class OMRReadResult:
    aligned_image: np.ndarray
    detections: dict[str, dict[str, Detection]]
    alignment: AlignmentResult

    def public_answers(self) -> dict[str, dict[str, str]]:
        return {
            part: {
                question: (
                    "Multiple"
                    if item.state == "multiple"
                    else "Uncertain"
                    if item.state == "uncertain"
                    else item.answer or "-"
                )
                for question, item in questions.items()
            }
            for part, questions in self.detections.items()
        }


class OMRProcessor:
    """Reads student marks only; it has no knowledge of correct answers."""

    def __init__(self, layout_path: str | Path | dict[str, Any] | None = None) -> None:
        self.layout = layout_path if isinstance(layout_path, dict) else load_layout(layout_path)

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise InvalidImageError("Unable to read this image.")
        return image

    def detect_bytes(self, data: bytes) -> OMRReadResult:
        image = self.decode_image(data)
        return self.detect_image(image)

    def detect_image(
        self, image: np.ndarray, registration_image: np.ndarray | None = None
    ) -> OMRReadResult:
        alignment = align_image(image, self.layout, registration_image)
        detections = detect_answers(alignment.image, self.layout)
        return OMRReadResult(alignment.image, detections, alignment)


@dataclass(frozen=True)
class AutoOMRReadResult:
    template_id: str
    template_name: str
    processor: OMRProcessor
    read: OMRReadResult


def _merge_discovered_rows(
    reference_layout: dict[str, Any], discovered_layout: dict[str, Any]
) -> dict[str, Any] | None:
    """Keep calibrated labels/options while following the rows visible in a scan."""
    layout = deepcopy(reference_layout)
    discovered_questions = list(iter_questions(discovered_layout))
    reference_by_part: dict[str, list[Any]] = {
        part: [] for part in reference_layout["parts"]
    }
    for question in iter_questions(reference_layout):
        reference_by_part.setdefault(question.part, []).append(question)

    for part, target in layout["parts"].items():
        options = target["options"]
        candidates = [q for q in discovered_questions if q.part == part]
        reference = reference_by_part.get(part, [])
        if not candidates:
            return None
        if any(len(q.bubbles) != len(options) for q in candidates):
            return None
        used_reference: set[str] = set()
        records: list[tuple[int | None, float, list[int], str | None]] = []
        for candidate in candidates:
            candidate_x = np.asarray([bubble.x for bubble in candidate.bubbles])
            best: tuple[float, Any] | None = None
            for question in reference:
                if question.question in used_reference:
                    continue
                reference_x = np.asarray([bubble.x for bubble in question.bubbles])
                if candidate_x.shape != reference_x.shape:
                    continue
                x_error = float(np.mean(np.abs(candidate_x - reference_x)))
                y_error = abs(candidate.bubbles[0].y - question.bubbles[0].y)
                cost = x_error + y_error * 0.15
                if x_error <= 12 and y_error <= 18 and (best is None or cost < best[0]):
                    best = (cost, question)
            if best is None:
                records.append(
                    (None, float(candidate.bubbles[0].y), [b.x for b in candidate.bubbles], None)
                )
            else:
                question = best[1]
                used_reference.add(question.question)
                ref_index = next(i for i, item in enumerate(reference) if item.question == question.question)
                records.append(
                    (
                        ref_index,
                        float(candidate.bubbles[0].y),
                        [b.x for b in candidate.bubbles],
                        question.question,
                    )
                )

        known = sorted((record for record in records if record[0] is not None), key=lambda row: row[0])
        extra = sorted((record for record in records if record[0] is None), key=lambda row: row[1])
        ordered = known + extra
        if not ordered:
            return None
        extra_index = 0
        grids: list[dict[str, Any]] = []
        for _, y, x_centers, label in ordered:
            if label is None:
                if part == "part1":
                    numeric = [int(q.question) for q in reference if q.question.isdigit()]
                    label = str(max(numeric, default=0) + extra_index + 1)
                else:
                    position = len(reference) + extra_index
                    label = f"2.{position // 8 + 1}{chr(ord('a') + position % 8)}"
                extra_index += 1
            grids.append(
                {
                    "question_labels": [label],
                    "count": 1,
                    "x_centers": x_centers,
                    "y_start": y,
                    "y_spacing": 1,
                }
            )
        target["grids"] = grids

    layout["id"] = "generic_dynamic"
    layout["name"] = "Automatically discovered bubble sheet"
    layout["alignment"] = {"enabled": False, "required": False}
    return layout


class AutoOMRProcessor:
    """Vision-test every installed layout and return the strongest match."""

    def __init__(self, layout_paths: list[Path] | None = None) -> None:
        paths = layout_paths if layout_paths is not None else discover_layout_paths()
        self.processors: dict[str, OMRProcessor] = {}
        for path in paths:
            processor = OMRProcessor(path)
            template_id = str(processor.layout["id"])
            if template_id in self.processors:
                raise ValueError(f"Duplicate OMR template id: {template_id}")
            self.processors[template_id] = processor
        if not self.processors:
            raise ValueError("No OMR layout definitions were found.")

    def detect_image(
        self,
        image: np.ndarray,
        registration_image: np.ndarray | None = None,
    ) -> AutoOMRReadResult:
        matches: list[AutoOMRReadResult] = []
        alignment_errors: list[AlignmentConfidenceError] = []
        for template_id, processor in self.processors.items():
            try:
                read = processor.detect_image(image, registration_image)
            except AlignmentConfidenceError as error:
                alignment_errors.append(error)
                continue
            except LayoutDetectionError:
                continue
            matches.append(
                AutoOMRReadResult(
                    template_id,
                    str(processor.layout.get("name", template_id)),
                    processor,
                    read,
                )
            )
        if matches:
            best = max(
                matches,
                key=lambda match: (
                    match.read.alignment.confidence,
                    match.read.alignment.matched_markers,
                ),
            )
            if best.processor.layout.get("alignment", {}).get("method") != "corner-marker-homography":
                return best
            try:
                discovered = detect_generic_layout(best.read.aligned_image)
            except LayoutDetectionError:
                return best
            template_counts = {
                part: len(questions)
                for part, questions in best.read.detections.items()
            }
            discovered_counts = {
                part: sum(
                    grid["count"]
                    for grid in discovered.layout["parts"][part]["grids"]
                )
                for part in discovered.layout["parts"]
            }
            if any(
                discovered_counts.get(part, 0) > template_counts.get(part, 0)
                for part in set(discovered_counts) | set(template_counts)
            ):
                discovered_layout = _merge_discovered_rows(
                    best.processor.layout, discovered.layout
                ) or discovered.layout
                processor = OMRProcessor(discovered_layout)
                detections = detect_answers(discovered.image, discovered_layout)
                return AutoOMRReadResult(
                    "generic_dynamic",
                    str(discovered_layout["name"]),
                    processor,
                    OMRReadResult(
                        discovered.image, detections, discovered.alignment
                    ),
                )
            return best

        if alignment_errors:
            # A known page can have valid markers but a different number or
            # arrangement of answer rows. Reuse its page coordinates to rectify
            # the photo, then discover the bubbles from the rectified image.
            references = sorted(
                (
                    processor
                    for processor in self.processors.values()
                    if processor.layout.get("alignment", {}).get("method")
                    == "corner-marker-homography"
                ),
                key=lambda processor: int(
                    processor.layout.get("alignment", {}).get(
                        "reference_priority", 0
                    )
                ),
                reverse=True,
            )
            for reference in references:
                try:
                    rectified = align_image(
                        image,
                        reference.layout,
                        registration_image,
                        validate_bubble_grid=False,
                    )
                    generic = detect_generic_layout(
                        rectified.image,
                    )
                    dynamic_layout = _merge_discovered_rows(
                        reference.layout, generic.layout
                    ) or generic.layout
                    processor = OMRProcessor(dynamic_layout)
                    detections = detect_answers(generic.image, dynamic_layout)
                    return AutoOMRReadResult(
                        "generic_dynamic",
                        str(dynamic_layout["name"]),
                        processor,
                        OMRReadResult(
                            generic.image, detections, generic.alignment
                        ),
                    )
                except LayoutDetectionError:
                    continue

        try:
            generic = detect_generic_layout(image, registration_image)
        except LayoutDetectionError:
            if alignment_errors:
                raise alignment_errors[0]
            raise
        processor = OMRProcessor(generic.layout)
        detections = detect_answers(generic.image, generic.layout)
        return AutoOMRReadResult(
            "generic_dynamic",
            str(generic.layout["name"]),
            processor,
            OMRReadResult(generic.image, detections, generic.alignment),
        )
