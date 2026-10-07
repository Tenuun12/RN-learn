from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alignment import AlignmentResult, align_image
from .config import load_layout
from .detector import Detection, detect_answers
from .errors import InvalidImageError


@dataclass(frozen=True)
class OMRReadResult:
    aligned_image: np.ndarray
    detections: dict[str, dict[str, Detection]]
    alignment: AlignmentResult

    def public_answers(self) -> dict[str, dict[str, str]]:
        return {
            part: {
                question: ("Multiple" if item.answer == "Multiple" else item.answer or "-")
                for question, item in questions.items()
            }
            for part, questions in self.detections.items()
        }


class OMRProcessor:
    """Reads student marks only; it has no knowledge of correct answers."""

    def __init__(self, layout_path: str | Path | None = None) -> None:
        self.layout = load_layout(layout_path)

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise InvalidImageError("Unable to read this image.")
        return image

    def detect_bytes(self, data: bytes) -> OMRReadResult:
        image = self.decode_image(data)
        alignment = align_image(image, self.layout)
        detections = detect_answers(alignment.image, self.layout)
        return OMRReadResult(alignment.image, detections, alignment)
