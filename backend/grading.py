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
    student_answers: dict[str, dict[str, str]] = {
        part: {
            question: (
                "Multiple"
                if detection.state == "multiple"
                else "Uncertain"
                if detection.state == "uncertain"
                else detection.answer or "-"
            )
            for question, detection in questions.items()
        }
        for part, questions in detections.items()
    }
    summaries: dict[str, Any] = {}
    aggregate = {
        "questions": 0,
        "total": 0,
        "key_total": 0,
        "detected": 0,
        "not_present": 0,
        "not_graded": 0,
        "correct": 0,
        "incorrect": 0,
        "unanswered": 0,
        "invalid": 0,
        "uncertain": 0,
        "points": 0.0,
    }

    for part in dict.fromkeys([*detections, *answer_key]):
        detected_answers = detections.get(part, {})
        keyed_answers = answer_key.get(part, {})
        grading[part] = {}
        counts = {
            "total": 0,
            "key_total": len(keyed_answers),
            "detected": len(detected_answers),
            "not_present": 0,
            "not_graded": 0,
            "correct": 0,
            "incorrect": 0,
            "unanswered": 0,
            "invalid": 0,
            "uncertain": 0,
            "points": 0.0,
        }
        questions = list(detected_answers)
        questions.extend(question for question in keyed_answers if question not in detected_answers)
        for question in questions:
            detection = detected_answers.get(question)
            correct_answer = keyed_answers.get(question)
            if detection is None:
                grading[part][question] = {
                    "student": "Not detected",
                    "correct_answer": correct_answer,
                    "status": "not-present",
                    "selected": [],
                    "confidence": 0.0,
                    "fill_scores": {},
                    "points": None,
                    "gradable": False,
                }
                counts["not_present"] += 1
                continue
            if correct_answer is None:
                display_answer = student_answers.get(part, {}).get(question, "-")
                grading[part][question] = {
                    "student": display_answer,
                    "correct_answer": None,
                    "status": "not-graded",
                    "selected": list(detection.selected),
                    "confidence": detection.confidence,
                    "fill_scores": detection.fill_scores,
                    "points": None,
                    "gradable": False,
                }
                counts["not_graded"] += 1
                continue
            if detection.state == "multiple":
                status = "invalid"
                display_answer = "Multiple"
            elif detection.state == "uncertain":
                status = "uncertain"
                display_answer = "Uncertain"
            elif detection.state == "blank":
                status = "unanswered"
                display_answer = "-"
            elif detection.answer == correct_answer:
                status = "correct"
                display_answer = detection.answer
            else:
                status = "incorrect"
                display_answer = detection.answer

            points = float(scoring.get(f"{status}_points", 0))
            counts["total"] += 1
            counts[status] += 1
            counts["points"] += points
            grading[part][question] = {
                "student": display_answer,
                "correct_answer": correct_answer,
                "status": status,
                "selected": list(detection.selected),
                "confidence": detection.confidence,
                "fill_scores": detection.fill_scores,
                "points": points,
                "gradable": True,
            }

        counts["score"] = round(100 * counts["correct"] / max(counts["total"], 1), 2)
        summaries[part] = counts
        aggregate["questions"] += counts["total"]
        aggregate["total"] += counts["total"]
        for field in ("key_total", "detected", "not_present", "not_graded"):
            aggregate[field] += counts[field]
        for field in (
            "correct",
            "incorrect",
            "unanswered",
            "invalid",
            "uncertain",
            "points",
        ):
            aggregate[field] += counts[field]

    aggregate["score"] = round(
        100 * aggregate["correct"] / max(aggregate["questions"], 1), 2
    )
    summaries["total"] = aggregate
    grading_coverage = {
        part: {
            "matched_questions": sum(
                row["gradable"] for row in questions.values()
            ),
            "detected_without_key": [
                question
                for question, row in questions.items()
                if row["status"] == "not-graded"
            ],
            "key_questions_not_detected": [
                question
                for question, row in questions.items()
                if row["status"] == "not-present"
            ],
        }
        for part, questions in grading.items()
    }
    return {
        "student_answers": student_answers,
        "grading": grading,
        "summary": summaries,
        "grading_coverage": grading_coverage,
    }
