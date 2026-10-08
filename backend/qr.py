from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

import cv2
import numpy as np


@dataclass(frozen=True)
class DecodedQRCode:
    data: str
    url: str | None
    corners: tuple[tuple[float, float], ...] = ()

    def as_dict(self) -> dict[str, str | None]:
        return {"data": self.data, "url": self.url}


@dataclass(frozen=True)
class QRDetectionResult:
    codes: tuple[DecodedQRCode, ...] = ()

    @property
    def detected(self) -> bool:
        return bool(self.codes)

    @property
    def first(self) -> DecodedQRCode | None:
        return self.codes[0] if self.codes else None

    def response_fields(self) -> dict[str, object]:
        first = self.first
        return {
            "qr_detected": self.detected,
            "qr_data": first.data if first else None,
            "qr_url": first.url if first else None,
            "qr_codes": [code.as_dict() for code in self.codes],
        }

    def mask_for_alignment(self, image: np.ndarray) -> np.ndarray:
        """Hide decoded QR finder patterns while retaining the original geometry."""
        masked = image.copy()
        for code in self.codes:
            if len(code.corners) < 4:
                continue
            polygon = np.float32(code.corners)
            center = polygon.mean(axis=0)
            expanded = center + (polygon - center) * 1.12
            cv2.fillConvexPoly(masked, np.int32(np.rint(expanded)), (255, 255, 255))
        return masked

    def preview_images(self, image: np.ndarray, maximum_side: int = 420) -> list[np.ndarray]:
        """Crop compact QR previews from the untouched upload for result display."""
        previews: list[np.ndarray] = []
        image_height, image_width = image.shape[:2]
        for code in self.codes:
            if len(code.corners) < 4:
                continue
            polygon = np.float32(code.corners)
            x, y, width, height = cv2.boundingRect(polygon)
            padding = max(12, int(round(max(width, height) * 0.12)))
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(image_width, x + width + padding)
            bottom = min(image_height, y + height + padding)
            if right <= left or bottom <= top:
                continue
            preview = image[top:bottom, left:right].copy()
            longest = max(preview.shape[:2])
            if longest > maximum_side:
                scale = maximum_side / longest
                preview = cv2.resize(
                    preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            previews.append(preview)
        return previews


def extract_http_url(data: str) -> str | None:
    """Return a safe, clickable HTTP(S) URL without changing its contents."""
    candidate = data.strip()
    if not candidate or any(character in candidate for character in "\r\n\t"):
        return None
    try:
        parsed = urlsplit(candidate)
        _ = parsed.port  # Validate malformed ports while parsing the URL.
    except (TypeError, ValueError):
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    return candidate


def _corners(points: np.ndarray | None) -> tuple[tuple[float, float], ...]:
    if points is None:
        return ()
    flattened = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return tuple((float(x), float(y)) for x, y in flattened)


def _decoded_values(
    detector: cv2.QRCodeDetector, image: np.ndarray
) -> list[tuple[str, tuple[tuple[float, float], ...]]]:
    values: list[tuple[str, tuple[tuple[float, float], ...]]] = []

    # Multi-code decoding is attempted first so no QR is silently discarded.
    try:
        detected, decoded, points, _straight = detector.detectAndDecodeMulti(image)
        if detected:
            for index, value in enumerate(decoded):
                if value:
                    code_points = points[index] if points is not None and index < len(points) else None
                    values.append((value, _corners(code_points)))
    except (cv2.error, TypeError, ValueError):
        pass

    # Some low-quality images work in the single-code path even when the multi
    # decoder found a boundary but could not decode it.
    try:
        decoded, points, _straight = detector.detectAndDecode(image)
        if decoded:
            values.append((decoded, _corners(points)))
    except (cv2.error, TypeError, ValueError):
        pass

    return values


def _detection_variants(image: np.ndarray) -> list[tuple[np.ndarray, float]]:
    height, width = image.shape[:2]
    longest = max(height, width)

    # Bound the working image on large phone photos to keep serverless memory
    # predictable while preserving enough detail for QR finder patterns.
    if longest > 2800:
        scale = 2800 / longest
        base = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        base = image

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base
    variants = [(base, 1 / scale), (gray, 1 / scale)]

    # Local contrast normalization helps faded or unevenly lit camera images.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append((clahe, 1 / scale))

    # Small QR codes need more pixels per module. Cap the expanded image at
    # roughly 12 MP to avoid a large transient allocation in Vercel functions.
    pixels = int(gray.shape[0] * gray.shape[1])
    if longest < 2200 and pixels * 4 <= 12_000_000:
        variants.append(
            (cv2.resize(clahe, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 1 / (scale * 2))
        )

    return variants


def detect_qr_codes(image: np.ndarray) -> QRDetectionResult:
    """Decode QR codes without allowing a QR failure to interrupt OMR grading."""
    if image is None or image.size == 0:
        return QRDetectionResult()

    found: dict[str, tuple[tuple[float, float], ...]] = {}
    try:
        detector = cv2.QRCodeDetector()
        for variant, coordinate_scale in _detection_variants(image):
            for value, corners in _decoded_values(detector, variant):
                original_corners = tuple(
                    (x * coordinate_scale, y * coordinate_scale) for x, y in corners
                )
                if value not in found or (not found[value] and original_corners):
                    found[value] = original_corners
    except (cv2.error, MemoryError, TypeError, ValueError):
        return QRDetectionResult()

    return QRDetectionResult(
        tuple(
            DecodedQRCode(data=value, url=extract_http_url(value), corners=corners)
            for value, corners in found.items()
        )
    )
