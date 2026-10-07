from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.omr.alignment import align_image
from backend.omr.config import load_layout
from backend.omr.detector import detect_answers
from backend.omr.errors import InvalidImageError
from backend.omr.processor import OMRProcessor
from backend.omr.visualizer import annotate_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw and label every configured OMR bubble.")
    parser.add_argument("image", type=Path, help="PNG or JPEG answer sheet")
    parser.add_argument("--output", type=Path, default=Path("debug_omr.png"))
    parser.add_argument("--layout", type=Path, default=None)
    args = parser.parse_args()

    try:
        image = OMRProcessor.decode_image(args.image.read_bytes())
    except (OSError, InvalidImageError) as exc:
        parser.error(str(exc))

    layout = load_layout(args.layout)
    aligned = align_image(image, layout)
    detections = detect_answers(aligned.image, layout)
    debug = annotate_calibration(aligned.image, layout, detections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), debug):
        parser.error(f"Unable to write {args.output}")
    print(
        f"Wrote {args.output} ({aligned.method}, "
        f"{aligned.matched_markers}/{aligned.expected_markers} registration markers matched)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

