# Teacher OMR Test Grader

A complete FastAPI, OpenCV, and React application for grading a student's OMR sheet against an answer key entered by a teacher.

```text
Teacher creates test and answer key
                 ↓
OpenCV reads student OMR marks
                 ↓
Independent grading layer compares both
                 ↓
Scores, answer table, and annotated sheet
```

There are **no hardcoded correct answers**. Tests and teacher keys are stored in `backend/data/tests.json`. The storage layer has a repository interface so it can later be replaced by a database without changing OMR detection or grading.

## Actual calibrated sheet

The layout was measured from `samples/filled.png`:

- Reference size: 1290 × 1344 pixels
- Part 1: 60 rows with choices A–E in two columns
- Part 2: numeric rows with choices `0–9`, `-`, and `.`
- Registration alignment: geometry-based affine transform using the printed black L-shaped marker lattice; it handles page margins, translation, rotation, independent X/Y scaling, and shear
- Bubble measurement: 7 px inner circular ROI; dark threshold 80; filled threshold 0.45; empty threshold 0.20

The supplied image is cropped after row `2.4f`, so the layout contains the 30 fully visible Part 2 rows. The answer-key UI uses the actual template options instead of pretending this is a generic A–D sheet.

## Project structure

```text
backend/
  main.py                   FastAPI test/key/grading routes
  storage.py                JSON test repository
  grading.py                answer comparison and scores
  calibrate.py              coordinate overlay CLI
  config/
    omr_layout.json         measured sheet geometry and thresholds
    scoring.json            configurable point rules
  data/tests.json           teacher-created tests and keys
  omr/
    alignment.py            registration-marker alignment
    detector.py             fill-ratio measurement
    processor.py            student-mark reading only
    visualizer.py           grading and calibration overlays
  tests/
frontend/
  src/App.jsx               teacher workflow
  src/styles.css
samples/filled.png
```

## Install and run

Use Python 3.11+ and Node.js 20+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements-dev.txt
python -m uvicorn backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. API documentation is at `http://localhost:8000/docs`.

## Teacher workflow

1. Select **New test**.
2. Enter a name and question counts for Parts 1 and 2.
3. Select exactly one correct response for every configured question.
4. Save the key. It remains editable through **Edit answer key**.
5. Upload a student's PNG/JPEG sheet or load the provided sample.
6. Select **Read OMR & grade**.
7. Review Part scores, teacher/student comparisons, confidence, and the colored detection overlay.

The test selector allows multiple named tests. Each student is graded against the currently selected test's saved key.

## API

### Create a test

```powershell
curl.exe -X POST http://localhost:8000/api/tests `
  -H "Content-Type: application/json" `
  -d '{"name":"Math Midterm","part_counts":{"part1":3,"part2":2}}'
```

### Save or edit its teacher answer key

For the actual sheet, the first two Part 2 row IDs are `2.1a` and `2.1b`.

```powershell
curl.exe -X POST http://localhost:8000/api/tests/TEST_ID/answer-key `
  -H "Content-Type: application/json" `
  -d '{"part1":{"1":"B","2":"B","3":"D"},"part2":{"2.1a":"8","2.1b":"6"}}'
```

The server rejects missing questions, extra questions, or options that do not exist on the calibrated template.

### Read the key and grade a student

```powershell
curl.exe http://localhost:8000/api/tests/TEST_ID/answer-key
curl.exe -X POST http://localhost:8000/api/tests/TEST_ID/grade -F "image=@samples/filled.png"
```

The grading response separates `answer_key`, `student_answers`, and `grading`, and includes summary counts plus `annotated_image_url`.

## Configuration

### Scoring

Edit `backend/config/scoring.json`. The current rules are one point for correct and zero for incorrect, unanswered, or multiple responses. New tests copy these settings when created.

### OMR thresholds and coordinates

Edit `backend/config/omr_layout.json`:

- `dark_pixel_threshold`: pixels below this grayscale value count as dark.
- `filled_threshold`: minimum dark-pixel ratio for a marked bubble.
- `empty_threshold`: upper boundary for a confidently empty bubble.
- `sample_radius`: inner circular measurement radius.
- grid `x_centers`, `y_start`, and `y_spacing`: calibrated positions.

On the sample, marked bubbles measure about 0.70–1.00 and empty bubbles approximately 0.00.

Photographed sheets also receive local illumination correction, a ±2 px bubble-center search, and a lower-confidence partial-mark rule. Darkness is calibrated per upload from the sheet's printed black registration markers, preventing washed-out camera images from turning marked answers into false blanks. The result screen reports how many registration markers supported the alignment. If it reports **Review alignment**, use a flatter, sharper photograph.

### Calibration overlay

```powershell
python backend\calibrate.py samples\filled.png --output debug_omr.png
```

The output labels every configured Part 1 and Part 2 bubble. Selected bubbles are green.

## Verification

```powershell
pytest -q
cd frontend
npm run build
```

Tests cover raw OMR reading without a key, teacher-controlled correct/incorrect grading, unanswered and multiple marks, answer-key validation, JSON persistence, the complete API workflow, scoring, annotated-image retrieval, and a translated/rotated/scaled/sheared/JPEG-compressed camera simulation whose 90 visible answers must all match the clean reference.
