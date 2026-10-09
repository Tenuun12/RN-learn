from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alignment import AlignmentResult, align_image
from .config import discover_layout_paths, load_layout
from .detector import Detection, detect_answers
from .errors import InvalidImageError, LayoutDetectionError
from .generic import detect_generic_layout


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
        question_labels: dict[str, list[str]] | None = None,
    ) -> AutoOMRReadResult:
        matches: list[AutoOMRReadResult] = []
        for template_id, processor in self.processors.items():
            try:
                read = processor.detect_image(image, registration_image)
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
            return max(
                matches,
                key=lambda match: (
                    match.read.alignment.confidence,
                    match.read.alignment.matched_markers,
                ),
            )

        generic = detect_generic_layout(image, registration_image, question_labels)
        processor = OMRProcessor(generic.layout)
        detections = detect_answers(generic.image, generic.layout)
        return AutoOMRReadResult(
            "generic_dynamic",
            str(generic.layout["name"]),
            processor,
            OMRReadResult(generic.image, detections, generic.alignment),
        )
