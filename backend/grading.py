from __future__ import annotations

from typing import Any

try:
    from backend.omr.detector import Detection
except ModuleNotFoundError:  # Vercel service root is backend/
    from omr.detector import Detection


def grade_detections(
    detections: dict[str, dict[str, Detection]],
    answer_key: dict[str, dict[str, str]],
    scoring: dict[str, Any],
) -> dict[str, Any]:
    grading: dict[str, dict[str, dict[str, Any]]] = {}
    student_answers: dict[str, dict[str, str]] = {}
    summaries: dict[str, Any] = {}
    aggregate = {
        "questions": 0,
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "unanswered": 0,
        "invalid": 0,
        "points": 0.0,
    }

    for part, keyed_answers in answer_key.items():
        grading[part] = {}
        student_answers[part] = {}
        counts = {
            "total": len(keyed_answers),
            "correct": 0,
            "incorrect": 0,
            "unanswered": 0,
            "invalid": 0,
            "points": 0.0,
        }
        for question, correct_answer in keyed_answers.items():
            detection = detections[part][question]
            if detection.answer == "Multiple":
                status = "invalid"
                display_answer = "Multiple"
            elif detection.answer is None:
                status = "unanswered"
                display_answer = "-"
            elif detection.answer == correct_answer:
                status = "correct"
                display_answer = detection.answer
            else:
                status = "incorrect"
                display_answer = detection.answer

            points = float(scoring.get(f"{status}_points", 0))
            counts[status] += 1
            counts["points"] += points
            student_answers[part][question] = display_answer
            grading[part][question] = {
                "student": display_answer,
                "correct_answer": correct_answer,
                "status": status,
                "selected": list(detection.selected),
                "confidence": detection.confidence,
                "fill_scores": detection.fill_scores,
                "points": points,
            }

        counts["score"] = round(100 * counts["correct"] / max(counts["total"], 1), 2)
        summaries[part] = counts
        aggregate["questions"] += counts["total"]
        aggregate["total"] += counts["total"]
        for field in ("correct", "incorrect", "unanswered", "invalid", "points"):
            aggregate[field] += counts[field]

    aggregate["score"] = round(
        100 * aggregate["correct"] / max(aggregate["questions"], 1), 2
    )
    summaries["total"] = aggregate
    return {
        "student_answers": student_answers,
        "grading": grading,
        "summary": summaries,
    }
