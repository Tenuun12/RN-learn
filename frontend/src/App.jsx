import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const EMPTY_KEY = { part1: {}, part2: {} };
const PART_TITLES = { part1: "Part 1", part2: "Part 2" };
const STATUS_LABELS = {
  correct: "Correct",
  incorrect: "Incorrect",
  unanswered: "Unanswered",
  invalid: "Multiple",
};

async function api(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => item.msg).join(" ")
      : payload.detail;
    throw new Error(detail || "The request could not be completed.");
  }
  return payload;
}

function StepRail({ stage }) {
  const current = { create: 1, key: 1, upload: 2, results: 3 }[stage] || 1;
  return (
    <div className="step-rail" aria-label={`Step ${current} of 3`}>
      {["Answer key", "Student OMR", "Results"].map((label, index) => {
        const step = index + 1;
        return (
          <div className={`step ${step === current ? "active" : ""} ${step < current ? "done" : ""}`} key={label}>
            <span>{step < current ? "✓" : step}</span>
            <div><small>Step {step}</small><strong>{label}</strong></div>
          </div>
        );
      })}
    </div>
  );
}

function CreateTest({ layout, onCreated, busy, setBusy, setError }) {
  const [name, setName] = useState("");
  const [counts, setCounts] = useState({ part1: 20, part2: 20 });

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const payload = await api("/api/tests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, part_counts: counts }),
      });
      onCreated(payload.test);
    } catch (error) {
      setError(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="workspace-card create-card">
      <div className="section-intro">
        <span className="eyebrow">Create test</span>
        <h1>Set up the answer key</h1>
        <p>Name the test and choose how many of the calibrated sheet rows belong to each part.</p>
      </div>
      <form onSubmit={submit}>
        <label className="field wide">
          <span>Test name</span>
          <input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Mathematics midterm" required maxLength={120} />
        </label>
        <div className="count-grid">
          {["part1", "part2"].map((part) => (
            <label className="field" key={part}>
              <span>{PART_TITLES[part]} questions</span>
              <input
                type="number"
                min="1"
                max={layout.parts[part].maximum_questions}
                value={counts[part]}
                onChange={(event) => setCounts({ ...counts, [part]: Number(event.target.value) })}
              />
              <small>Maximum {layout.parts[part].maximum_questions} on this template</small>
            </label>
          ))}
        </div>
        <div className="form-footer">
          <div className="secure-note"><span>01</span> Teacher answers become the only grading source of truth.</div>
          <button className="primary-button" disabled={busy || !name.trim()}>{busy ? "Creating…" : "Create test & enter answers"}<b>→</b></button>
        </div>
      </form>
    </section>
  );
}

function PartKeyEditor({ part, schema, values, onChange }) {
  const answered = schema.question_labels.filter((question) => values[question]).length;
  const firstMissing = schema.question_labels.find((question) => !values[question]);
  return (
    <section className="key-part">
      <div className="key-part__header">
        <div><span className="eyebrow">{PART_TITLES[part]}</span><h2>Correct answers</h2></div>
        <div className={`progress-count ${answered === schema.question_count ? "complete" : ""}`}>
          {answered}/{schema.question_count}
        </div>
      </div>
      <div className="progress-track"><i style={{ width: `${100 * answered / schema.question_count}%` }} /></div>
      <div className="question-list">
        {schema.question_labels.map((question, index) => (
          <div className={`answer-row ${values[question] ? "answered" : ""}`} key={question}>
            <div className="question-label"><small>Question</small><strong>{part === "part2" ? `${index + 1} · ${question}` : question}</strong></div>
            <div className="option-group" role="radiogroup" aria-label={`${PART_TITLES[part]} question ${question}`}>
              {schema.options.map((option) => (
                <button
                  type="button"
                  className={values[question] === option ? "selected" : ""}
                  aria-pressed={values[question] === option}
                  onClick={() => onChange(question, option)}
                  key={option}
                >{option}</button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className={`validation-line ${firstMissing ? "pending" : "valid"}`}>
        <span>{firstMissing ? "!" : "✓"}</span>
        {firstMissing ? `${PART_TITLES[part]} question ${firstMissing} has no correct answer.` : `${PART_TITLES[part]}: all ${schema.question_count} answers entered.`}
      </p>
    </section>
  );
}

function AnswerKeyEditor({ test, initialKey, onSaved, busy, setBusy, setError }) {
  const [key, setKey] = useState(() => ({
    part1: { ...(initialKey?.part1 || {}) },
    part2: { ...(initialKey?.part2 || {}) },
  }));
  const complete = ["part1", "part2"].every((part) =>
    test.parts[part].question_labels.every((question) => key[part][question]),
  );

  function update(part, question, answer) {
    setKey((current) => ({ ...current, [part]: { ...current[part], [question]: answer } }));
  }

  async function save() {
    if (!complete) return;
    setBusy(true);
    setError("");
    try {
      const payload = await api(`/api/tests/${test.id}/answer-key`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(key),
      });
      onSaved(payload.test, payload.answer_key);
    } catch (error) {
      setError(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="editor-shell">
      <div className="section-intro compact">
        <span className="eyebrow">{initialKey ? "Edit answer key" : "Create answer key"}</span>
        <h1>{test.name}</h1>
        <p>Select exactly one correct answer for every configured row. Options mirror the actual calibrated OMR sheet.</p>
      </div>
      <div className="key-grid">
        {["part1", "part2"].map((part) => (
          <PartKeyEditor
            key={part}
            part={part}
            schema={test.parts[part]}
            values={key[part]}
            onChange={(question, answer) => update(part, question, answer)}
          />
        ))}
      </div>
      <div className="sticky-actions">
        <div><strong>{complete ? "Answer key complete" : "Complete every question to continue"}</strong><span>{complete ? "Ready to save and grade students." : "Missing answers are marked beneath each part."}</span></div>
        <button className="primary-button" disabled={!complete || busy} onClick={save}>{busy ? "Saving…" : initialKey ? "Save changes" : "Save answer key"}<b>→</b></button>
      </div>
    </section>
  );
}

function UploadStage({ test, onEdit, onGraded, busy, setBusy, setError }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!file) { setPreview(""); return undefined; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function chooseFile(candidate) {
    if (!candidate) return;
    if (!candidate.type.match(/^image\/(png|jpe?g)$/)) {
      setError("Please choose a PNG or JPG/JPEG image.");
      return;
    }
    setFile(candidate);
    setError("");
  }

  async function loadSample() {
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/sample`);
      if (!response.ok) throw new Error("Unable to load the provided sample.");
      chooseFile(new File([await response.blob()], "filled.png", { type: "image/png" }));
    } catch (error) { setError(error.message); }
  }

  async function grade() {
    if (!file) { setError("Please upload a student's OMR answer sheet."); return; }
    setBusy(true);
    setError("");
    const body = new FormData();
    body.append("image", file);
    try {
      const payload = await api(`/api/tests/${test.id}/grade`, { method: "POST", body });
      onGraded(payload);
    } catch (error) {
      setError(error.message === "Failed to fetch" ? "Cannot reach the OMR server." : error.message);
    } finally { setBusy(false); }
  }

  return (
    <section className="workspace-card upload-card">
      <div className="upload-heading">
        <div><span className="eyebrow">Student answer sheet</span><h1>{test.name}</h1><p>The saved teacher key is locked in for this grading run.</p></div>
        <button className="secondary-button" onClick={onEdit}>Edit answer key</button>
      </div>
      <div className="key-summary">
        <span><b>✓</b> Answer key saved</span>
        <span>Part 1 · {test.part_counts.part1} questions</span>
        <span>Part 2 · {test.part_counts.part2} questions</span>
      </div>
      <div
        className={`drop-zone ${dragging ? "dragging" : ""} ${preview ? "has-preview" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]); }}
      >
        {preview ? (
          <div className="preview-layout">
            <img src={preview} alt="Selected student OMR preview" />
            <div><span className="ready-icon">✓</span><span className="eyebrow">Ready to grade</span><h2>{file.name}</h2><p>{(file.size / 1024 / 1024).toFixed(2)} MB · PNG/JPEG</p><button className="text-button" onClick={() => inputRef.current?.click()}>Choose a different image</button></div>
          </div>
        ) : (
          <button className="empty-drop" onClick={() => inputRef.current?.click()}><span className="upload-icon">↑</span><strong>Drop the student's OMR sheet here</strong><span>or click to choose a PNG or JPEG</span></button>
        )}
        <input ref={inputRef} type="file" accept="image/png,image/jpeg" hidden onChange={(event) => chooseFile(event.target.files[0])} />
      </div>
      {!file && <button className="sample-button" onClick={loadSample}>Use provided sample sheet</button>}
      <button className="primary-button grade-button" disabled={!file || busy} onClick={grade}>{busy ? <><i className="spinner" />Reading bubbles…</> : <>Read OMR & grade <b>→</b></>}</button>
    </section>
  );
}

function ScoreCard({ title, summary, tone }) {
  return (
    <article className={`score-card ${tone || ""}`}>
      <div><span className="eyebrow">{title}</span><strong>{summary.correct}<small> / {summary.total}</small></strong></div>
      <div className="score-ring" style={{ "--score": `${summary.score * 3.6}deg` }}><span>{summary.score}%</span></div>
      <div className="score-stats"><span><i className="dot correct" />{summary.correct} correct</span><span><i className="dot incorrect" />{summary.incorrect} incorrect</span><span><i className="dot unanswered" />{summary.unanswered} blank</span><span><i className="dot invalid" />{summary.invalid} multiple</span></div>
    </article>
  );
}

function AnswerTable({ part, rows }) {
  return (
    <section className="answer-panel">
      <div className="panel-heading"><div><span className="eyebrow">Answer comparison</span><h2>{PART_TITLES[part]}</h2></div><span className="pill">{Object.keys(rows).length} questions</span></div>
      <div className="table-wrap"><table><thead><tr><th>Question</th><th>Teacher key</th><th>Student OMR</th><th>Result</th><th>Confidence</th></tr></thead><tbody>
        {Object.entries(rows).map(([question, item]) => <tr key={question}><td className="question">{question}</td><td><span className="answer-chip teacher">{item.correct_answer}</span></td><td><span className="answer-chip">{item.student}</span></td><td><span className={`status ${item.status}`}><i />{STATUS_LABELS[item.status]}</span></td><td className="confidence">{Math.round(item.confidence * 100)}%</td></tr>)}
      </tbody></table></div>
    </section>
  );
}

function Results({ data, onAnother, onEdit }) {
  const annotatedUrl = `${API_BASE}${data.annotated_image_url}`;
  return (
    <section className="results">
      <div className="results-heading"><div><span className="eyebrow">Grading complete</span><h1>{data.test.name}</h1><p>Teacher key compared against detected student marks.</p><span className={`alignment-detail ${data.alignment.confidence < 0.7 ? "review" : ""}`}>{data.alignment.matched_markers}/{data.alignment.expected_markers} registration markers aligned · {data.alignment.confidence >= 0.7 ? "High-confidence fit" : "Review alignment"}</span></div><div className="heading-actions"><button className="secondary-button" onClick={onEdit}>Edit key</button><button className="primary-button" onClick={onAnother}>Grade another student</button></div></div>
      <div className="score-grid"><ScoreCard title="Part 1" summary={data.summary.part1} /><ScoreCard title="Part 2" summary={data.summary.part2} tone="blue" /><ScoreCard title="Total score" summary={data.summary.total} tone="dark" /></div>
      <section className="visual-panel"><div className="panel-heading"><div><span className="eyebrow">Detection overlay</span><h2>Visual verification</h2></div><div className="legend"><span className="correct">Correct</span><span className="incorrect">Incorrect</span><span className="unanswered">Blank</span><span className="invalid">Multiple</span></div></div><a href={annotatedUrl} target="_blank" rel="noreferrer"><img src={annotatedUrl} alt="Student OMR with grading annotations" /></a></section>
      <AnswerTable part="part1" rows={data.grading.part1} />
      <AnswerTable part="part2" rows={data.grading.part2} />
    </section>
  );
}

export default function App() {
  const [layout, setLayout] = useState(null);
  const [tests, setTests] = useState([]);
  const [activeTest, setActiveTest] = useState(null);
  const [answerKey, setAnswerKey] = useState(null);
  const [stage, setStage] = useState("create");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api("/api/layout"), api("/api/tests")])
      .then(async ([layoutPayload, testsPayload]) => {
        setLayout(layoutPayload);
        setTests(testsPayload.tests);
        if (testsPayload.tests.length) await selectTest(testsPayload.tests[0]);
      })
      .catch((loadError) => setError(loadError.message));
  }, []);

  async function selectTest(test) {
    setActiveTest(test);
    setResult(null);
    setError("");
    if (test.has_answer_key) {
      try {
        const payload = await api(`/api/tests/${test.id}/answer-key`);
        setAnswerKey(payload.answer_key);
        setStage("upload");
      } catch (loadError) { setError(loadError.message); }
    } else {
      setAnswerKey(null);
      setStage("key");
    }
  }

  function created(test) {
    setTests((current) => [test, ...current]);
    setActiveTest(test);
    setAnswerKey(null);
    setStage("key");
  }

  function saved(test, key) {
    setActiveTest(test);
    setAnswerKey(key);
    setTests((current) => current.map((item) => item.id === test.id ? test : item));
    setStage("upload");
  }

  const headerTitle = useMemo(() => activeTest?.name || "Teacher grading studio", [activeTest]);

  if (!layout) return <main className="loading-screen"><span className="spinner dark-spinner" /><p>Loading calibrated OMR workspace…</p>{error && <div className="error-banner">{error}</div>}</main>;

  return (
    <main>
      <header className="app-header">
        <div className="topbar"><div className="brand"><span>OM</span><div><strong>OMR Teacher</strong><small>Grading workspace</small></div></div><div className="test-switcher">{tests.length > 0 && <select aria-label="Select test" value={activeTest?.id || ""} onChange={(event) => selectTest(tests.find((test) => test.id === event.target.value))}>{tests.map((test) => <option value={test.id} key={test.id}>{test.name}</option>)}</select>}<button onClick={() => { setActiveTest(null); setAnswerKey(null); setResult(null); setStage("create"); }}>+ New test</button></div></div>
        <div className="header-copy"><span className="kicker">Teacher-owned answer keys</span><h2>{headerTitle}</h2><p>Create the key, scan the student's sheet, then review every detected mark.</p></div>
        <StepRail stage={stage} />
      </header>
      <div className="page-shell">
        {error && <div className="error-banner" role="alert"><span>!</span>{error}<button onClick={() => setError("")}>×</button></div>}
        {stage === "create" && <CreateTest layout={layout} onCreated={created} busy={busy} setBusy={setBusy} setError={setError} />}
        {stage === "key" && activeTest && <AnswerKeyEditor test={activeTest} initialKey={answerKey} onSaved={saved} busy={busy} setBusy={setBusy} setError={setError} />}
        {stage === "upload" && activeTest && <UploadStage test={activeTest} onEdit={() => setStage("key")} onGraded={(payload) => { setResult(payload); setStage("results"); window.scrollTo({ top: 0, behavior: "smooth" }); }} busy={busy} setBusy={setBusy} setError={setError} />}
        {stage === "results" && result && <Results data={result} onAnother={() => setStage("upload")} onEdit={() => setStage("key")} />}
      </div>
      <footer>OMR Teacher <span>·</span> Teacher key → OMR detection → grading <span>·</span> OpenCV</footer>
    </main>
  );
}
