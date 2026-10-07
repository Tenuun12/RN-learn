from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from backend.grading import grade_detections
    from backend.omr.errors import InvalidImageError, LayoutDetectionError
    from backend.omr.layout import questions_by_part
    from backend.omr.processor import OMRProcessor
    from backend.omr.visualizer import annotate_results
    from backend.storage import AnswerKeyNotFoundError, TestNotFoundError, TestRepository
except ModuleNotFoundError:  # Vercel service root is backend/
    from grading import grade_detections
    from omr.errors import InvalidImageError, LayoutDetectionError
    from omr.layout import questions_by_part
    from omr.processor import OMRProcessor
    from omr.visualizer import annotate_results
    from storage import AnswerKeyNotFoundError, TestNotFoundError, TestRepository


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RUNTIME_DIR = Path(
    os.getenv(
        "OMR_RUNTIME_DIR",
        str(Path(tempfile.gettempdir()) / "omr-teacher" / "results")
        if os.getenv("VERCEL")
        else str(BASE_DIR / "runtime"),
    )
)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}


class CreateTestRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    part_counts: dict[str, int]


class AnswerKeyRequest(BaseModel):
    part1: dict[str, str]
    part2: dict[str, str]


processor = OMRProcessor()
repository = TestRepository()
with (BASE_DIR / "config" / "scoring.json").open("r", encoding="utf-8") as handle:
    default_scoring: dict[str, Any] = json.load(handle)


def _part_schema() -> dict[str, dict[str, Any]]:
    return {
        part: {
            "label": processor.layout["parts"][part]["label"],
            "options": processor.layout["parts"][part]["options"],
            "question_labels": [question.question for question in questions],
            "maximum_questions": len(questions),
        }
        for part, questions in questions_by_part(processor.layout).items()
    }


def _serialize_test(record: dict[str, Any]) -> dict[str, Any]:
    schema = _part_schema()
    parts = {}
    for part in ("part1", "part2"):
        count = record["part_counts"][part]
        parts[part] = {
            **schema[part],
            "question_count": count,
            "question_labels": schema[part]["question_labels"][:count],
        }
    return {
        **{key: value for key, value in record.items() if key != "answer_key"},
        "has_answer_key": record["answer_key"] is not None,
        "parts": parts,
    }


def _validate_part_counts(part_counts: dict[str, int]) -> dict[str, int]:
    if set(part_counts) != {"part1", "part2"}:
        raise HTTPException(status_code=422, detail="Part 1 and Part 2 question counts are required.")
    schema = _part_schema()
    for part, count in part_counts.items():
        maximum = schema[part]["maximum_questions"]
        if count < 1 or count > maximum:
            label = schema[part]["label"]
            raise HTTPException(
                status_code=422,
                detail=f"{label} must contain between 1 and {maximum} questions for this OMR template.",
            )
    return part_counts


def _validate_answer_key(
    test: dict[str, Any], payload: AnswerKeyRequest
) -> dict[str, dict[str, str]]:
    submitted = {"part1": payload.part1, "part2": payload.part2}
    schema = _part_schema()
    errors: list[str] = []
    cleaned: dict[str, dict[str, str]] = {}
    for part in ("part1", "part2"):
        count = test["part_counts"][part]
        expected = schema[part]["question_labels"][:count]
        allowed = set(schema[part]["options"])
        provided = submitted[part]
        missing = [question for question in expected if question not in provided]
        extras = [question for question in provided if question not in expected]
        invalid = [
            question
            for question in expected
            if question in provided and provided[question] not in allowed
        ]
        if missing:
            errors.append(f"{schema[part]['label']} is missing: {', '.join(missing)}.")
        if extras:
            errors.append(f"{schema[part]['label']} has unexpected questions: {', '.join(extras)}.")
        if invalid:
            errors.append(
                f"{schema[part]['label']} has invalid answers for: {', '.join(invalid)}. "
                f"Allowed answers: {', '.join(schema[part]['options'])}."
            )
        cleaned[part] = {question: provided[question] for question in expected if question in provided}
    if errors:
        raise HTTPException(status_code=422, detail=" ".join(errors))
    return cleaned


app = FastAPI(title="Teacher OMR Test Grader", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/layout")
def layout_metadata() -> dict[str, Any]:
    return {"name": processor.layout["name"], "parts": _part_schema()}


@app.get("/api/sample")
def sample_sheet() -> FileResponse:
    service_sample = BASE_DIR / "samples" / "filled.png"
    path = service_sample if service_sample.exists() else PROJECT_ROOT / "samples" / "filled.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample sheet not found.")
    return FileResponse(path, media_type="image/png", filename="filled.png")


@app.get("/api/tests")
def list_tests() -> dict[str, Any]:
    return {"tests": [_serialize_test(record) for record in repository.list_tests()]}


@app.post("/api/tests", status_code=201)
def create_test(payload: CreateTestRequest) -> dict[str, Any]:
    counts = _validate_part_counts(payload.part_counts)
    record = repository.create_test(payload.name, counts, default_scoring)
    return {"success": True, "test": _serialize_test(record)}


@app.get("/api/tests/{test_id}")
def get_test(test_id: str) -> dict[str, Any]:
    try:
        return {"test": _serialize_test(repository.get_test(test_id))}
    except TestNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Test not found.") from exc


@app.post("/api/tests/{test_id}/answer-key")
def save_answer_key(test_id: str, payload: AnswerKeyRequest = Body(...)) -> dict[str, Any]:
    try:
        test = repository.get_test(test_id)
    except TestNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Test not found.") from exc
    answer_key = _validate_answer_key(test, payload)
    updated = repository.save_answer_key(test_id, answer_key)
    return {"success": True, "test": _serialize_test(updated), "answer_key": answer_key}


@app.get("/api/tests/{test_id}/answer-key")
def get_answer_key(test_id: str) -> dict[str, Any]:
    try:
        answer_key = repository.get_answer_key(test_id)
    except TestNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Test not found.") from exc
    except AnswerKeyNotFoundError as exc:
        raise HTTPException(status_code=404, detail="This test does not have an answer key yet.") from exc
    return {"test_id": test_id, "answer_key": answer_key}


@app.post("/api/tests/{test_id}/grade")
async def grade_student_sheet(test_id: str, image: UploadFile = File(...)) -> dict[str, Any]:
    try:
        test = repository.get_test(test_id)
        answer_key = repository.get_answer_key(test_id)
    except TestNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Test not found.") from exc
    except AnswerKeyNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail="Create and save the teacher answer key before grading a student.",
        ) from exc

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Please upload a PNG or JPG/JPEG image.")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Please upload a student's OMR answer sheet.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image must be 15 MB or smaller.")

    try:
        omr = processor.detect_bytes(data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LayoutDetectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    graded = grade_detections(omr.detections, answer_key, test["scoring"])
    annotated = annotate_results(omr.aligned_image, processor.layout, graded["grading"])
    result_id = uuid4().hex
    output_path = RUNTIME_DIR / f"{result_id}.png"
    if not cv2.imwrite(str(output_path), annotated):
        raise HTTPException(status_code=500, detail="Unable to create the annotated result image.")
    encoded_ok, encoded_overlay = cv2.imencode(
        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    if not encoded_ok:
        raise HTTPException(status_code=500, detail="Unable to encode the annotated result image.")
    annotated_data_url = (
        "data:image/jpeg;base64,"
        + base64.b64encode(encoded_overlay.tobytes()).decode("ascii")
    )

    return {
        "success": True,
        "test": _serialize_test(test),
        "answer_key": answer_key,
        **graded,
        "alignment": {
            "method": omr.alignment.method,
            "matched_markers": omr.alignment.matched_markers,
            "expected_markers": omr.alignment.expected_markers,
            "confidence": round(omr.alignment.confidence, 4),
        },
        "result_id": result_id,
        "annotated_image_url": f"/api/results/{result_id}/annotated.png",
        "annotated_image_data_url": annotated_data_url,
    }


@app.get("/api/results/{result_id}/annotated.png")
def annotated_result(result_id: str) -> FileResponse:
    if not result_id.isalnum() or len(result_id) != 32:
        raise HTTPException(status_code=404, detail="Result not found.")
    path = RUNTIME_DIR / f"{result_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")
    return FileResponse(path, media_type="image/png", filename="annotated_result.png")
