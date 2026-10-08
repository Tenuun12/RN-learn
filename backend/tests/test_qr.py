from __future__ import annotations

import cv2
import numpy as np

from backend.qr import detect_qr_codes, extract_http_url


def make_qr(data: str, size: int = 320) -> np.ndarray:
    encoded = cv2.QRCodeEncoder_create().encode(data)
    return cv2.resize(encoded, (size, size), interpolation=cv2.INTER_NEAREST)


def on_canvas(qr: np.ndarray, *, rotate: bool = False) -> np.ndarray:
    if rotate:
        qr = cv2.rotate(qr, cv2.ROTATE_90_CLOCKWISE)
    canvas = np.full((680, 880, 3), 245, dtype=np.uint8)
    height, width = qr.shape[:2]
    canvas[170 : 170 + height, 280 : 280 + width] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    return canvas


def test_url_qr_is_detected_and_returned_exactly() -> None:
    expected = "https://example.com/results/student-42?attempt=1#score"
    result = detect_qr_codes(on_canvas(make_qr(expected), rotate=True))

    assert result.detected is True
    assert result.first is not None
    assert result.first.data == expected
    assert result.first.url == expected
    assert result.response_fields()["qr_url"] == expected


def test_plain_text_qr_has_data_but_no_url() -> None:
    result = detect_qr_codes(on_canvas(make_qr("Student ID: MN-2048")))

    assert result.detected is True
    assert result.first is not None
    assert result.first.data == "Student ID: MN-2048"
    assert result.first.url is None


def test_multiple_qr_codes_are_returned_without_duplicates() -> None:
    first = make_qr("https://example.com/one", 260)
    second = make_qr("plain text two", 260)
    canvas = np.full((600, 900, 3), 255, dtype=np.uint8)
    canvas[150:410, 80:340] = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    canvas[150:410, 560:820] = cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)

    result = detect_qr_codes(canvas)

    assert {code.data for code in result.codes} == {
        "https://example.com/one",
        "plain text two",
    }
    assert len(result.codes) == 2


def test_no_qr_and_damaged_qr_are_non_fatal() -> None:
    blank = np.full((680, 880, 3), 255, dtype=np.uint8)
    assert detect_qr_codes(blank).response_fields() == {
        "qr_detected": False,
        "qr_data": None,
        "qr_url": None,
        "qr_codes": [],
    }

    damaged = make_qr("https://example.com/should-not-decode")
    damaged[:, : damaged.shape[1] * 3 // 4] = 255
    result = detect_qr_codes(on_canvas(damaged))
    assert result.detected is False


def test_only_http_and_https_payloads_become_links() -> None:
    assert extract_http_url("https://example.com/a") == "https://example.com/a"
    assert extract_http_url("HTTP://example.com/a") == "HTTP://example.com/a"
    assert extract_http_url("javascript:alert(1)") is None
    assert extract_http_url("not a URL") is None
