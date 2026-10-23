from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

try:
    from backend.grading import grade_detections
    from backend.omr.errors import InvalidImageError, LayoutDetectionError
    from backend.omr.layout import questions_by_part
    from backend.omr.processor import AutoOMRProcessor, OMRProcessor
    from backend.omr.visualizer import annotate_results
    from backend.qr import QRDetectionResult, detect_qr_codes, read_printed_student_id
    from backend.storage import AnswerKeyNotFoundError, TestNotFoundError, TestRepository
except ModuleNotFoundError:  # Vercel service root is backend/
    from grading import grade_detections
    from omr.errors import InvalidImageError, LayoutDetectionError
    from omr.layout import questions_by_part
    from omr.processor import AutoOMRProcessor, OMRProcessor
    from omr.visualizer import annotate_results
    from qr import QRDetectionResult, detect_qr_codes, read_printed_student_id
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
DEFAULT_PART_COUNTS = {"part1": 40, "part2": 32}


class CreateTestRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    part_counts: dict[str, int] = Field(
        default_factory=lambda: dict(DEFAULT_PART_COUNTS)
    )
    layout_id: str = "auto"


class AnswerKeyRequest(BaseModel):
    part1: dict[str, str]
    part2: dict[str, str]


AUTO_LAYOUT_ID = "auto"
auto_processor = AutoOMRProcessor()
processors = auto_processor.processors
repository = TestRepository()
with (BASE_DIR / "config" / "scoring.json").open("r", encoding="utf-8") as handle:
    default_scoring: dict[str, Any] = json.load(handle)


def _processor_for_layout(layout_id: str) -> OMRProcessor:
    selected = processors.get(layout_id)
    if selected is None:
        raise HTTPException(status_code=422, detail="Unknown answer-sheet template.")
    return selected


def _automatic_part_schema() -> dict[str, dict[str, Any]]:
    """Combine installed layouts so test setup does not require choosing one."""
    combined: dict[str, dict[str, Any]] = {}
    for selected in processors.values():
        for part, schema in _part_schema(str(selected.layout["id"])).items():
            target = combined.setdefault(
                part,
                {
                    "label": schema["label"],
                    "options": [],
                    "question_labels": [],
                    "maximum_questions": 0,
                },
            )
            for option in schema["options"]:
                if option not in target["options"]:
                    target["options"].append(option)
            for question in schema["question_labels"]:
                if question not in target["question_labels"]:
                    target["question_labels"].append(question)
            target["maximum_questions"] = max(
                target["maximum_questions"], schema["maximum_questions"]
            )
    return combined


def _part_schema(layout_id: str = AUTO_LAYOUT_ID) -> dict[str, dict[str, Any]]:
    if layout_id == AUTO_LAYOUT_ID:
        return _automatic_part_schema()
    selected = _processor_for_layout(layout_id)
    return {
        part: {
            "label": selected.layout["parts"][part]["label"],
            "options": selected.layout["parts"][part]["options"],
            "question_labels": [question.question for question in questions],
            "maximum_questions": len(questions),
        }
        for part, questions in questions_by_part(selected.layout).items()
    }


def _serialize_test(record: dict[str, Any]) -> dict[str, Any]:
    layout_id = record.get("layout_id", AUTO_LAYOUT_ID)
    schema = _part_schema(layout_id)
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
        "layout_id": layout_id,
        "has_answer_key": record["answer_key"] is not None,
        "parts": parts,
    }


def _validate_part_counts(
    part_counts: dict[str, int], layout_id: str = AUTO_LAYOUT_ID
) -> dict[str, int]:
    if set(part_counts) != {"part1", "part2"}:
        raise HTTPException(status_code=422, detail="Part 1 and Part 2 question counts are required.")
    schema = _part_schema(layout_id)
    for part, count in part_counts.items():
        maximum = schema[part]["maximum_questions"]
        if count < 0 or count > maximum:
            label = schema[part]["label"]
            raise HTTPException(
                status_code=422,
                detail=f"{label} must contain between 0 and {maximum} questions.",
            )
    return part_counts


def _validate_answer_key(
    test: dict[str, Any], payload: AnswerKeyRequest
) -> dict[str, dict[str, str]]:
    submitted = {"part1": payload.part1, "part2": payload.part2}
    schema = _part_schema(test.get("layout_id", AUTO_LAYOUT_ID))
    errors: list[str] = []
    cleaned: dict[str, dict[str, str]] = {}
    for part in ("part1", "part2"):
        count = test["part_counts"][part]
        expected = schema[part]["question_labels"][:count]
        allowed = set(schema[part]["options"])
        provided = submitted[part]
        extras = [question for question in provided if question not in expected]
        invalid = [
            question
            for question in expected
            if question in provided and provided[question] not in allowed
        ]
        if extras:
            errors.append(f"{schema[part]['label']} has unexpected questions: {', '.join(extras)}.")
        if invalid:
            errors.append(
                f"{schema[part]['label']} has invalid answers for: {', '.join(invalid)}. "
                f"Allowed answers: {', '.join(schema[part]['options'])}."
            )
        cleaned[part] = {question: provided[question] for question in expected if question in provided}
    if not any(cleaned[part] for part in ("part1", "part2")):
        errors.append("Select at least one answer before saving the answer key.")
    if errors:
        raise HTTPException(status_code=422, detail=" ".join(errors))
    return cleaned


def _serialize_detections(detections: dict[str, Any]) -> dict[str, Any]:
    """Expose every question and every dynamically available option."""
    result: dict[str, Any] = {}
    for part, questions in detections.items():
        result[part] = {}
        for question, detection in questions.items():
            if detection.state == "multiple":
                answer = "Multiple"
            elif detection.state == "uncertain":
                answer = "Uncertain"
            else:
                answer = detection.answer or "-"
            result[part][question] = {
                "answer": answer,
                "state": detection.state,
                "selected": list(detection.selected),
                "confidence": detection.confidence,
                "options": [
                    {
                        "label": option,
                        "fill_score": score,
                        "selected": option in detection.selected,
                    }
                    for option, score in detection.fill_scores.items()
                ],
            }
    return result


def _numeric_answer(answer: str | None, option_labels: str) -> str:
    """Map a selected bubble label to its 1-based numeric API representation."""
    if answer is None or answer in {"-", "Multiple", "Uncertain"}:
        return "0"
    # Part 2's numeric bubbles retain their printed 0-9 values.
    if option_labels == "0123456789" and answer in "0123456789":
        return answer
    if answer in option_labels:
        return str(option_labels.index(answer) + 1)
    return "0"


def _paper_test_id(qr_payload: str | None) -> str | None:
    """Read the paper ID from supported QR payloads."""
    if not qr_payload:
        return None
    explicit = re.search(r"(?:^|[|?&])paper\s*[:=]\s*(\d+)", qr_payload, re.IGNORECASE)
    if explicit:
        return str(int(explicit.group(1)))
    rn_format = re.search(r"(?:^|[^A-Z0-9])RN-(\d+)-\d+(?:$|[^0-9])", qr_payload, re.IGNORECASE)
    if rn_format:
        return str(int(rn_format.group(1)))
    return None


def _external_result(
    *,
    qr_payload: str | None,
    file_url: str,
    variant: str | None,
    student_answers: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Serialize the grading result in the consumer's problem/subproblem format."""
    test_user_id = qr_payload
    if test_user_id:
        test_user_id = re.split(r"\|paper\s*[:=]", test_user_id, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    questions: list[dict[str, str]] = []
    for question_id, answer in student_answers.get("part1", {}).items():
        questions.append(
            {
                "problem_type": "1",
                "problem_id": str(question_id),
                "problem_sub_id": "0",
                "answer": _numeric_answer(answer, "ABCDE"),
            }
        )

    # Preserve the sheet's printed group and subquestion labels (for example,
    # 2.1a -> problem 1, subproblem 1; 2.4h -> problem 4, subproblem 8).
    for index, (question_id, answer) in enumerate(student_answers.get("part2", {}).items()):
        label_match = re.fullmatch(r"2\.(\d+)([a-z])", question_id.strip(), re.IGNORECASE)
        if label_match:
            problem_id = int(label_match.group(1))
            problem_sub_id = ord(label_match.group(2).lower()) - ord("a") + 1
        else:
            # Keep unexpected labels visible in a stable eight-row order.
            problem_id = index // 8 + 1
            problem_sub_id = index % 8 + 1
        questions.append(
            {
                "problem_type": "2",
                "problem_id": str(problem_id),
                "problem_sub_id": str(problem_sub_id),
                "answer": _numeric_answer(answer, "0123456789"),
            }
        )

    return {
        "paper_test_id": _paper_test_id(qr_payload),
        "test_user_id": test_user_id,
        "file_url": file_url,
        "variant": _numeric_answer(variant, "ABCD") if variant else "0",
        "questions": questions,
    }


app = FastAPI(
    title="Teacher OMR Test Grader",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


@app.get("/docs", include_in_schema=False)
def local_swagger_docs() -> RedirectResponse:
    """Keep the conventional local FastAPI docs URL as an alias."""
    return RedirectResponse(url="/api/docs", status_code=307)


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


@app.get("/api")
def api_index() -> dict[str, Any]:
    return {
        "name": app.title,
        "version": app.version,
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc",
            "openapi": "/api/openapi.json",
        },
        "endpoints": [
            {"method": "GET", "path": "/api/health"},
            {"method": "GET", "path": "/api/layout"},
            {"method": "GET", "path": "/api/sample"},
            {"method": "GET", "path": "/api/tests"},
            {"method": "POST", "path": "/api/tests"},
            {"method": "GET", "path": "/api/tests/{test_id}"},
            {"method": "GET", "path": "/api/tests/{test_id}/answer-key"},
            {"method": "POST", "path": "/api/tests/{test_id}/answer-key"},
            {"method": "POST", "path": "/api/tests/{test_id}/grade"},
            {"method": "GET", "path": "/api/results/{result_id}/annotated.png"},
        ],
    }


@app.get("/api/layout")
def layout_metadata() -> dict[str, Any]:
    templates = [
        {
            "id": layout_id,
            "name": selected.layout["name"],
            "parts": _part_schema(layout_id),
        }
        for layout_id, selected in processors.items()
    ]
    return {
        "id": AUTO_LAYOUT_ID,
        "name": "Automatic answer-sheet detection",
        "mode": "automatic",
        "default_part_counts": dict(DEFAULT_PART_COUNTS),
        "parts": _part_schema(AUTO_LAYOUT_ID),
        "templates": templates,
    }


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
    # Explicit layout ids from older clients remain valid for answer-key editing,
    # but new clients use automatic detection and grading always detects visually.
    layout_id = payload.layout_id or AUTO_LAYOUT_ID
    counts = _validate_part_counts(payload.part_counts, layout_id)
    record = repository.create_test(
        payload.name, counts, default_scoring, layout_id
    )
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
async def grade_student_sheet(
    test_id: str,
    include_details: bool = Query(
        False,
        description="Set true to include grading, alignment, and image debug details.",
    ),
    image: UploadFile = File(...),
    answer_key_json: str | None = Form(None),
    test_config_json: str | None = Form(None),
) -> dict[str, Any]:
    if bool(answer_key_json) != bool(test_config_json):
        raise HTTPException(
            status_code=422,
            detail="The saved answer key and test configuration must be provided together.",
        )

    if answer_key_json and test_config_json:
        try:
            config = json.loads(test_config_json)
            key_payload = AnswerKeyRequest.model_validate_json(answer_key_json)
            name = str(config["name"]).strip()
            layout_id = str(config.get("layout_id") or AUTO_LAYOUT_ID)
            counts = _validate_part_counts(
                config.get("part_counts", DEFAULT_PART_COUNTS), layout_id
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="The saved answer key is invalid.") from exc
        if not name:
            raise HTTPException(status_code=422, detail="The saved test name is required.")
        test = {
            "id": test_id,
            "name": name[:120],
            "layout_id": layout_id,
            "part_counts": counts,
            "answer_key": None,
            "scoring": default_scoring,
            "created_at": config.get("created_at", ""),
            "updated_at": config.get("updated_at", ""),
        }
        answer_key = _validate_answer_key(test, key_payload)
    else:
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
        source_image = OMRProcessor.decode_image(data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # QR scanning is deliberately isolated from OMR processing: unreadable,
    # damaged, or unsupported QR content must never prevent a valid grade.
    try:
        qr_result = detect_qr_codes(source_image)
    except Exception:  # A third-party decoder failure is non-fatal by contract.
        qr_result = QRDetectionResult()

    try:
        registration_image = qr_result.mask_for_alignment(source_image)
        automatic_match = auto_processor.detect_image(
            source_image,
            registration_image,
        )
        selected_processor = automatic_match.processor
        omr = automatic_match.read
    except LayoutDetectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    qr_payload = qr_result.first.data if qr_result.first else None
    student_id_match = (
        re.search(r"RN-\d{4}-\d{4}", qr_payload, re.IGNORECASE)
        if qr_payload
        else None
    )
    student_id = student_id_match.group(0).upper() if student_id_match else None
    student_id_source = "qr" if student_id else None
    if student_id is None:
        layout = selected_processor.layout
        reference_size = (
            int(layout["image_width"]),
            int(layout["image_height"]),
        ) if layout.get("image_width") and layout.get("image_height") else None
        student_id = read_printed_student_id(
            omr.aligned_image,
            layout.get("student_id_region"),
            reference_size,
        )
        if student_id:
            student_id_source = "printed-text-ocr"
    qr_result = QRDetectionResult(
        qr_result.codes,
        student_id,
        student_id_source,
    )

    graded = grade_detections(omr.detections, answer_key, test["scoring"])
    annotated = annotate_results(
        omr.aligned_image,
        selected_processor.layout,
        graded["grading"],
        omr.detections,
    )
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
    qr_image_data_urls: list[str] = []
    for preview in qr_result.preview_images(source_image):
        preview_ok, encoded_preview = cv2.imencode(".png", preview)
        if preview_ok:
            qr_image_data_urls.append(
                "data:image/png;base64,"
                + base64.b64encode(encoded_preview.tobytes()).decode("ascii")
            )

    variant_detection = omr.fields.get("variant") if omr.fields else None
    variant_info = (
        None
        if variant_detection is None
        else {
            "answer": variant_detection.answer,
            "state": variant_detection.state,
            "selected": list(variant_detection.selected),
            "confidence": variant_detection.confidence,
        }
    )
    annotated_image_url = f"/api/results/{result_id}/annotated.png"
    qr_payload = qr_result.first.data if qr_result.first else qr_result.student_id
    result_payload = _external_result(
        qr_payload=qr_payload,
        file_url=annotated_image_url,
        variant=variant_detection.answer if variant_detection else None,
        student_answers=graded["student_answers"],
    )

    if not include_details:
        return {"result": result_payload}

    return {
        "success": True,
        "test": _serialize_test(test),
        "answer_key": answer_key,
        "detected_answers": _serialize_detections(omr.detections),
        "variant": variant_info,
        "result": result_payload,
        **graded,
        **qr_result.response_fields(),
        "alignment": {
            "template_id": automatic_match.template_id,
            "template_name": automatic_match.template_name,
            "selection": "automatic",
            "method": omr.alignment.method,
            "matched_markers": omr.alignment.matched_markers,
            "expected_markers": omr.alignment.expected_markers,
            "confidence": round(omr.alignment.confidence, 4),
        },
        "result_id": result_id,
        "annotated_image_url": annotated_image_url,
        "annotated_image_data_url": annotated_data_url,
        "qr_image_data_url": qr_image_data_urls[0] if qr_image_data_urls else None,
        "qr_image_data_urls": qr_image_data_urls,
    }


@app.get("/api/results/{result_id}/annotated.png")
def annotated_result(result_id: str) -> FileResponse:
    if not result_id.isalnum() or len(result_id) != 32:
        raise HTTPException(status_code=404, detail="Result not found.")
    path = RUNTIME_DIR / f"{result_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")
    return FileResponse(path, media_type="image/png", filename="annotated_result.png")
