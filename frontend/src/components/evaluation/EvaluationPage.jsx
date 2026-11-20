import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Wordmark } from '../Icons.jsx';
import Plot, { SERIES } from './Plot.jsx';
import {
  BinarySection,
  Card,
  ClassificationSection,
  DetectionSection,
  Empty,
  fmtDate,
  HistoricalSection,
  Info,
  InferenceSection,
  Metric,
  ms,
  num,
  pct,
} from './sections.jsx';
import { TASKS, cancelJob, compareRuns, deleteRun, exportUrl, getHistory, getJob, getRun, getSummary, saveRun, startEvaluation } from '../../lib/evalApi.js';
import '../../styles/evaluation.css';

const IMAGE_RE = /\.(jpe?g|png|webp|bmp)$/i;
const AUX_RE = /\.(txt|csv|ya?ml)$/i;

const FORMAT_HELP = {
  classification: (
    <>
      <code>dataset/plastic/img001.jpg</code>, <code>dataset/glass/…</code> — one folder per class (ewaste / e-waste, glass, hazardous, metal, organic,
      paper / paper_cardboard, plastic) — or a <code>labels.csv</code> with <code>filename,label</code>.
    </>
  ),
  binary: (
    <>
      <code>dataset/waste/…</code> and <code>dataset/non_waste/…</code> (folders named after a waste category also count as waste), or a <code>labels.csv</code>.
    </>
  ),
  detection: (
    <>
      Ultralytics YOLO layout: <code>images/x.jpg</code> + <code>labels/x.txt</code> (<code>class cx cy w h</code>, normalised) and a <code>data.yaml</code>{' '}
      with <code>names</code>.
    </>
  ),
  inference: <>Any JPG / PNG / WebP images. Labels are ignored — only timing is measured.</>,
};

function useSummary() {
  const [state, setState] = useState({ status: 'loading', data: null, error: null });
  const load = useCallback(async () => {
    try {
      const data = await getSummary();
      setState({ status: 'ok', data, error: null });
    } catch (e) {
      setState({ status: 'error', data: null, error: e.message });
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);
  return { ...state, reload: load };
}

// ------------------------------------------------------------------ run form
function RunForm({ summary, onRun, busy, setBusy }) {
  const caps = summary?.capabilities || {};
  const defaults = summary?.defaults || {};
  const [taskId, setTaskId] = useState('classification');
  const task = TASKS.find((t) => t.id === taskId);
  const [files, setFiles] = useState([]);
  const [name, setName] = useState('');
  const [nameTouched, setNameTouched] = useState(false);
  const [params, setParams] = useState({});
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const zipRef = useRef(null);
  const dirRef = useRef(null);
  const imgRef = useRef(null);
  const jobRef = useRef(null);

  // reset parameters to the backend defaults when the task changes
  useEffect(() => {
    const d = defaults[task.endpoint] || {};
    setParams({ conf: d.conf, iou: d.iou, max_det: d.max_det, warmup: d.warmup ?? 1, gradcam: d.gradcam ?? true });
  }, [taskId, summary]); // eslint-disable-line react-hooks/exhaustive-deps

  const pick = (list, fromDir) => {
    const arr = Array.from(list || []);
    const zips = arr.filter((f) => /\.zip$/i.test(f.name));
    let chosen;
    if (zips.length && !fromDir) chosen = [{ file: zips[0], path: zips[0].name }];
    else chosen = arr.filter((f) => IMAGE_RE.test(f.name) || AUX_RE.test(f.name)).map((f) => ({ file: f, path: f.webkitRelativePath || f.name }));
    setFiles(chosen);
    setError(chosen.length ? null : 'No ZIP or image files were selected.');
    if (chosen.length && !nameTouched) {
      const p = chosen[0].path;
      setName(p.includes('/') ? p.split('/')[0] : p.replace(/\.zip$/i, ''));
    }
  };

  const summaryText = useMemo(() => {
    if (!files.length) return null;
    if (files.length === 1 && /\.zip$/i.test(files[0].file.name)) return `ZIP · ${(files[0].file.size / 1048576).toFixed(1)} MB`;
    const n = files.filter((f) => IMAGE_RE.test(f.file.name)).length;
    const mb = files.reduce((a, f) => a + f.file.size, 0) / 1048576;
    return `${n} image${n === 1 ? '' : 's'}${files.length > n ? ` + ${files.length - n} label/config file(s)` : ''} · ${mb.toFixed(1)} MB`;
  }, [files]);

  const start = async () => {
    setError(null);
    setBusy(true);
    setProgress({ phase: 'upload', frac: 0 });
    try {
      const body = { dataset_name: name, warmup: params.warmup };
      if (task.params.includes('conf')) Object.assign(body, { conf: params.conf, iou: params.iou, max_det: params.max_det });
      if (task.params.includes('gradcam')) body.gradcam = params.gradcam ? 'true' : 'false';
      const { job_id } = await startEvaluation(task, files, body, (frac) => setProgress({ phase: 'upload', frac }));
      jobRef.current = job_id;
      // poll
      for (;;) {
        await new Promise((r) => setTimeout(r, 600));
        const j = await getJob(job_id);
        setProgress({ phase: j.status, frac: j.total ? j.done / j.total : 0, job: j });
        if (j.status === 'done') {
          onRun(await getRun(j.run_id));
          break;
        }
        if (j.status === 'error') throw new Error(j.error?.message || 'Evaluation failed.');
        if (j.status === 'cancelled') throw new Error('Evaluation cancelled.');
      }
      setProgress(null);
    } catch (e) {
      setError(e.message);
      setProgress(null);
    } finally {
      jobRef.current = null;
      setBusy(false);
    }
  };

  const capOk = caps[task.capability];
  const set = (k) => (e) => setParams((p) => ({ ...p, [k]: e.target.type === 'checkbox' ? e.target.checked : e.target.value === '' ? '' : Number(e.target.value) }));

  return (
    <Card title="Run an evaluation" eyebrow="Dataset evaluation workflow" id="ev-run" wide>
      <fieldset className="ev-tasks" disabled={busy}>
        <legend className="ev-sub">1 · Evaluation task</legend>
        {TASKS.map((t) => (
          <label key={t.id} className={`ev-task ${taskId === t.id ? 'is-on' : ''} ${caps[t.capability] ? '' : 'is-off'}`}>
            <input type="radio" name="task" value={t.id} checked={taskId === t.id} onChange={() => setTaskId(t.id)} disabled={!caps[t.capability]} />
            <span className="ev-task__title">{t.title}</span>
            <span className="ev-task__blurb">{caps[t.capability] ? t.blurb : 'Unavailable: the required model is not loaded on the server.'}</span>
          </label>
        ))}
      </fieldset>

      <div className="ev-form-grid">
        <div>
          <p className="ev-sub">2 · Dataset</p>
          <div
            className={`ev-drop ${drag ? 'is-drag' : ''}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              pick(e.dataTransfer.files, false);
            }}
          >
            <p className="ev-drop__title">{summaryText || 'Drop a dataset ZIP here'}</p>
            <div className="ev-drop__btns">
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => zipRef.current?.click()} disabled={busy}>
                Choose ZIP
              </button>
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => dirRef.current?.click()} disabled={busy}>
                Choose folder
              </button>
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => imgRef.current?.click()} disabled={busy}>
                Choose images
              </button>
            </div>
            <p className="ev-drop__help">{FORMAT_HELP[task.format]}</p>
          </div>
          <input ref={zipRef} type="file" accept=".zip,application/zip" className="sr-only" onChange={(e) => (pick(e.target.files, false), (e.target.value = ''))} />
          <input ref={dirRef} type="file" webkitdirectory="" directory="" multiple className="sr-only" onChange={(e) => (pick(e.target.files, true), (e.target.value = ''))} />
          <input ref={imgRef} type="file" accept="image/jpeg,image/png,image/webp,.bmp" multiple className="sr-only" onChange={(e) => (pick(e.target.files, false), (e.target.value = ''))} />
          <label className="ev-field">
            <span>Dataset name</span>
            <input type="text" value={name} maxLength={120} onChange={(e) => (setName(e.target.value), setNameTouched(true))} placeholder="e.g. test-split-v1" disabled={busy} />
          </label>
        </div>

        <div>
          <p className="ev-sub">3 · Parameters</p>
          {task.params.includes('conf') ? (
            <>
              <label className="ev-field">
                <span>
                  YOLO confidence threshold <Info text={task.id === 'detection' ? 'Ultralytics validation uses 0.001 so the full precision–recall curve is measured. The live website uses 0.05.' : 'The live website uses 0.05.'} />
                </span>
                <input type="number" min="0.001" max="0.99" step="0.001" value={params.conf ?? ''} onChange={set('conf')} disabled={busy} />
              </label>
              <label className="ev-field">
                <span>
                  NMS IoU threshold <Info text="IoU used by YOLO's non-maximum suppression. The live website uses 0.45. mAP matching always uses IoU 0.50–0.95." />
                </span>
                <input type="number" min="0.05" max="0.95" step="0.05" value={params.iou ?? ''} onChange={set('iou')} disabled={busy} />
              </label>
              <label className="ev-field">
                <span>Max detections per image</span>
                <input type="number" min="1" max="300" step="1" value={params.max_det ?? ''} onChange={set('max_det')} disabled={busy} />
              </label>
            </>
          ) : (
            <p className="ev-note small">
              {task.id === 'binary'
                ? `Binary threshold is fixed at ${summary?.models?.binary_filter?.threshold ?? 0.35} (the value used by the live pipeline).`
                : 'No YOLO parameters: the classifier sees each full image.'}
            </p>
          )}
          {task.params.includes('gradcam') && (
            <label className="ev-check">
              <input type="checkbox" checked={!!params.gradcam} onChange={set('gradcam')} disabled={busy} /> Include Grad-CAM in the timing
            </label>
          )}
          <label className="ev-field">
            <span>
              Warm-up images <Info text="Images run once before timing starts, so one-off start-up cost is not counted." />
            </span>
            <input type="number" min="0" max="10" step="1" value={params.warmup ?? 1} onChange={set('warmup')} disabled={busy} />
          </label>
        </div>
      </div>

      <div className="ev-run-actions">
        {!busy ? (
          <button type="button" className="btn btn--primary" onClick={start} disabled={!files.length || !capOk}>
            Start evaluation
          </button>
        ) : (
          <>
            <div className="ev-progress" role="status" aria-live="polite">
              <div className="ev-progress__bar">
                <span style={{ width: `${Math.round((progress?.frac || 0) * 100)}%` }} />
              </div>
              <p className="small mono">
                {progress?.phase === 'upload'
                  ? `Uploading… ${Math.round((progress.frac || 0) * 100)}%`
                  : progress?.job
                    ? `${progress.job.status === 'queued' ? 'Queued' : 'Evaluating'} ${progress.job.done}/${progress.job.total || '?'} · ${progress.job.elapsed_s}s ${progress.job.current ? `· ${progress.job.current}` : ''}`
                    : 'Starting…'}
              </p>
            </div>
            {progress?.job && (
              <button type="button" className="btn btn--ghost btn--sm" onClick={() => jobRef.current && cancelJob(jobRef.current)}>
                Cancel
              </button>
            )}
          </>
        )}
      </div>
      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}
      <p className="ev-note small">The upload is processed in a temporary folder on the server and deleted when the run ends. Only the results are kept — and only if you save the run.</p>
    </Card>
  );
}

// ------------------------------------------------------------------ run header + KPIs
function RunHeader({ run, onSaved }) {
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);
  const save = async () => {
    setSaving(true);
    setErr(null);
    try {
      await saveRun(run.run_id, run.dataset.display_name || run.dataset.name);
      onSaved();
    } catch (e) {
      setErr(e.message);
    } finally {
      setSaving(false);
    }
  };
  const c = run.classification;
  const ds = run.dataset;
  return (
    <div className="ev-runhead">
      <div className="ev-runhead__meta">
        <p className="ev-badge ev-badge--measured">Measured in this {run.saved ? 'saved run' : 'session'}</p>
        <h2 className="ev-runhead__title">{run.task_label}</h2>
        <p className="muted small">
          {ds.display_name || ds.name} · {fmtDate(run.created_at)} · run <span className="mono">{run.run_id}</span> · labels: {ds.label_format}
        </p>
      </div>
      <div className="ev-runhead__actions">
        {run.saved ? (
          <span className="ev-badge">Saved</span>
        ) : (
          <button type="button" className="btn btn--ghost btn--sm" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save run'}
          </button>
        )}
        <a className="btn btn--primary btn--sm" href={exportUrl(run.run_id, 'pdf')} download>
          PDF report
        </a>
        <a className="btn btn--ghost btn--sm" href={exportUrl(run.run_id, 'csv')} download>
          All metrics CSV
        </a>
        {err && <span className="error-text small">{err}</span>}
      </div>
      <div className="ev-kpis">
        <Metric label="Evaluated images" value={ds.n_images_evaluated ?? '—'} sub={ds.n_invalid ? `${ds.n_invalid} invalid file(s) skipped` : `${ds.n_images_found} found`} />
        <Metric
          label="Detected objects"
          value={run.objects_detected ?? '—'}
          sub={run.objects_detected === null || run.objects_detected === undefined ? 'YOLO not used in this task' : run.pipeline_counts ? `${run.pipeline_counts.rejected_by_binary_gate} rejected by binary gate` : 'YOLO candidate boxes'}
        />
        <Metric
          label="Classification accuracy"
          value={c ? pct(c.accuracy) : '—'}
          sub={c ? `macro F1 ${num(c.macro_avg.f1)}` : run.supervised === false ? 'Not measured — needs labels' : 'Not part of this run'}
        />
        <Metric label="Avg inference time" value={ms(run.inference?.end_to_end?.mean)} sub={run.inference?.fps ? `${num(run.inference.fps, 2)} FPS · see stages below` : ''} />
      </div>
      {run.warnings?.length > 0 && (
        <ul className="ev-warn" role="status">
          {run.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
      {ds.invalid_files?.length > 0 && (
        <details className="ev-details">
          <summary>{ds.n_invalid} invalid file(s)</summary>
          <ul className="small mono">
            {ds.invalid_files.slice(0, 50).map((f) => (
              <li key={f.file}>
                {f.file} — {f.reason}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function PerImageTable({ run }) {
  const rows = run?.per_image || [];
  if (!rows.length) return null;
  const keys = Object.keys(rows[0]).filter((k) => k !== 'timing_ms');
  return (
    <details className="ev-details ev-card ev-card--wide">
      <summary>Per-image results ({rows.length})</summary>
      <div className="ev-table-wrap ev-table-wrap--tall">
        <table className="ev-table ev-table--left small">
          <thead>
            <tr>
              {keys.map((k) => (
                <th key={k} scope="col">
                  {k.replace(/_/g, ' ')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 300).map((r, i) => (
              <tr key={i}>
                {keys.map((k) => (
                  <td key={k} className="mono">
                    {Array.isArray(r[k]) ? r[k].join(', ') : r[k] === null || r[k] === undefined ? '—' : String(r[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <a className="btn btn--ghost btn--sm" href={exportUrl(run.run_id, 'csv', 'per_image')} download>
        Per-image CSV
      </a>
    </details>
  );
}

// ------------------------------------------------------------------ history + compare
function HistorySection({ runs, currentId, onOpen, onChanged }) {
  const [sel, setSel] = useState([]);
  const [cmp, setCmp] = useState(null);
  const [err, setErr] = useState(null);
  const toggle = (id) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  const doCompare = async () => {
    setErr(null);
    try {
      setCmp((await compareRuns(sel)).runs);
    } catch (e) {
      setErr(e.message);
    }
  };
  const remove = async (id) => {
    try {
      await deleteRun(id);
      setSel((s) => s.filter((x) => x !== id));
      onChanged();
    } catch (e) {
      setErr(e.message);
    }
  };
  const metrics = [
    ['accuracy', 'Accuracy'],
    ['macro_f1', 'Macro F1'],
    ['binary_accuracy', 'Binary acc.'],
    ['map50', 'mAP@0.50'],
    ['map50_95', 'mAP@.50:.95'],
  ];
  const cmpData = useMemo(() => {
    if (!cmp) return null;
    const present = metrics.filter(([k]) => cmp.some((r) => r[k] !== null && r[k] !== undefined));
    return cmp.slice(0, 8).map((r, i) => ({
      type: 'bar',
      name: `${r.dataset_name} · ${r.run_id.slice(0, 6)}`,
      x: present.map(([, l]) => l),
      y: present.map(([k]) => r[k]),
      marker: { color: SERIES[i % SERIES.length] },
      hovertemplate: `${r.dataset_name}<br>%{x}: %{y:.3f}<extra></extra>`,
    }));
  }, [cmp]); // eslint-disable-line react-hooks/exhaustive-deps
  const cmpLayout = useMemo(() => ({ barmode: 'group', yaxis: { range: [0, 1.02], title: 'Score' } }), []);

  return (
    <Card
      title="Evaluation history"
      eyebrow="Saved and recent runs"
      id="ev-history"
      wide
      actions={
        <button type="button" className="btn btn--ghost btn--sm" disabled={sel.length < 2} onClick={doCompare}>
          Compare selected ({sel.length})
        </button>
      }
    >
      {!runs.length ? (
        <Empty>No runs yet. Unsaved runs disappear when the server restarts; save a run to keep its results.</Empty>
      ) : (
        <div className="ev-table-wrap">
          <table className="ev-table ev-table--left">
            <thead>
              <tr>
                <th scope="col">
                  <span className="sr-only">Select</span>
                </th>
                <th scope="col">Date</th>
                <th scope="col">Dataset</th>
                <th scope="col">Task</th>
                <th scope="col">Images</th>
                <th scope="col">Checkpoints (sha256)</th>
                <th scope="col">Accuracy</th>
                <th scope="col">F1 (macro)</th>
                <th scope="col">mAP@0.50</th>
                <th scope="col">Avg time</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className={r.run_id === currentId ? 'is-current' : ''}>
                  <td>
                    <input type="checkbox" checked={sel.includes(r.run_id)} onChange={() => toggle(r.run_id)} aria-label={`Select run ${r.run_id}`} />
                  </td>
                  <td className="small">{fmtDate(r.created_at)}</td>
                  <td>
                    {r.dataset_name} {!r.saved && <span className="ev-badge ev-badge--dim">unsaved</span>}
                  </td>
                  <td className="small">{r.task_label}</td>
                  <td>{r.n_images}</td>
                  <td className="mono small" title={JSON.stringify(r.models)}>
                    {Object.values(r.models)
                      .filter(Boolean)
                      .map((h) => h.slice(0, 8))
                      .join(' · ')}
                  </td>
                  <td>{r.accuracy !== null && r.accuracy !== undefined ? pct(r.accuracy) : r.binary_accuracy !== null && r.binary_accuracy !== undefined ? `${pct(r.binary_accuracy)} (bin)` : '—'}</td>
                  <td>{num(r.macro_f1 ?? r.binary_f1)}</td>
                  <td>{num(r.map50)}</td>
                  <td>{ms(r.avg_inference_ms)}</td>
                  <td>
                    <div className="ev-row-actions">
                    <button type="button" className="link-btn" onClick={() => onOpen(r.run_id)}>
                      Open
                    </button>
                    {r.saved && (
                      <button type="button" className="link-btn" onClick={() => remove(r.run_id)}>
                        Delete
                      </button>
                    )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {err && <p className="error-text">{err}</p>}
      {cmp && (
        <div className="ev-compare">
          <p className="ev-sub">Comparison</p>
          {cmpData?.[0]?.x.length ? <Plot data={cmpData} layout={cmpLayout} height={300} label="Comparison of selected runs" /> : <Empty>The selected runs have no common supervised metrics.</Empty>}
          <div className="ev-table-wrap">
            <table className="ev-table">
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  {cmp.map((r) => (
                    <th key={r.run_id} scope="col">
                      {r.dataset_name}
                      <br />
                      <span className="mono small">{r.run_id}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[['task_label', 'Task'], ['n_images', 'Images'], ...metrics, ['binary_f1', 'Binary F1'], ['det_precision', 'Det. precision'], ['det_recall', 'Det. recall'], ['avg_inference_ms', 'Avg time (ms)'], ['fps', 'FPS']].map(([k, l]) => (
                  <tr key={k}>
                    <th scope="row">{l}</th>
                    {cmp.map((r) => (
                      <td key={r.run_id}>{typeof r[k] === 'number' ? (k === 'n_images' ? r[k] : k === 'avg_inference_ms' || k === 'fps' ? r[k].toFixed(1) : r[k].toFixed(3)) : r[k] ?? '—'}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {cmp.some((r) => r.task !== cmp[0].task) && <p className="ev-note small">These runs use different tasks; compare like with like.</p>}
          {new Set(cmp.map((r) => JSON.stringify(r.models))).size > 1 && <p className="ev-note small">The runs used different model checkpoints (see sha256).</p>}
        </div>
      )}
    </Card>
  );
}

// ------------------------------------------------------------------ page
function ModelConfig({ summary }) {
  const m = summary?.models;
  if (!m) return null;
  const row = (label, x, extra) => (
    <li>
      <span className="ev-cfg__label">{label}</span>
      <span className={`status-dot status-dot--${x?.loaded ? 'ready' : 'offline'}`} aria-hidden="true" />
      <span className="mono small">{x?.checkpoint?.file || '—'}</span>
      <span className="muted small mono">{x?.checkpoint?.sha256}</span>
      {extra && <span className="muted small">{extra}</span>}
    </li>
  );
  return (
    <ul className="ev-cfg" aria-label="Loaded models">
      {row('YOLOv8 detector', m.detector, `conf ${m.detector.inference_confidence} · IoU ${m.detector.inference_iou}`)}
      {row('Binary MobileNetV3', m.binary_filter, `threshold ${m.binary_filter.threshold}`)}
      {row('Seven-class MobileNetV3', m.classifier, `${m.classifier.input_size}px · uncertain < ${m.classifier.uncertainty_threshold}`)}
      <li>
        <span className="ev-cfg__label">Device</span>
        <span className="mono small">{m.device}</span>
      </li>
    </ul>
  );
}

export default function EvaluationPage() {
  const summary = useSummary();
  const [run, setRun] = useState(null);
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);
  const [openErr, setOpenErr] = useState(null);
  const resultsRef = useRef(null);

  const loadHistory = useCallback(async () => {
    try {
      setHistory((await getHistory()).runs);
    } catch {
      /* summary error is shown instead */
    }
  }, []);
  useEffect(() => {
    loadHistory();
    window.scrollTo(0, 0);
    document.title = 'Performance Evaluation · EcoSortAI';
    return () => {
      document.title = 'EcoSortAI';
    };
  }, [loadHistory]);

  const onRun = (r) => {
    setRun(r);
    loadHistory();
    requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  };
  const open = async (id) => {
    setOpenErr(null);
    try {
      onRun(await getRun(id));
    } catch (e) {
      setOpenErr(e.message);
    }
  };
  const refreshCurrent = async () => {
    if (run) setRun(await getRun(run.run_id));
    loadHistory();
  };

  const s = summary.data;
  return (
    <div className="ev-page">
      <header className="ev-top">
        <a className="topbar__brand" href="#/" aria-label="EcoSortAI — back to the journey">
          <Wordmark />
        </a>
        <nav className="ev-top__nav" aria-label="Site">
          <a href="#/" className="ev-back">
            ← Journey &amp; Waste Lab
          </a>
          <span className="ev-top__here" aria-current="page">
            Performance Evaluation
          </span>
        </nav>
      </header>

      <main className="ev-main">
        <div className="ev-hero">
          <p className="eyebrow">Performance Evaluation</p>
          <h1 className="ev-hero__title">
            How well does EcoSortAI <em>actually</em> work?
          </h1>
          <p className="ev-hero__lede">
            Evaluate the complete system — YOLOv8 detection, the binary waste gate, the seven-class classifier and the inference pipeline — on your own labelled
            test data. Every number on this page is computed from real predictions against ground truth; nothing is estimated or pre-filled.
          </p>
          {summary.status === 'error' && (
            <div className="error-box" role="alert">
              <p className="error-box__title">Evaluation API unavailable: {summary.error}</p>
              <button type="button" className="btn btn--ghost btn--sm" onClick={summary.reload}>
                Retry
              </button>
            </div>
          )}
          {s && <ModelConfig summary={s} />}
        </div>

        {s && <RunForm summary={s} onRun={onRun} busy={busy} setBusy={setBusy} />}
        {openErr && <p className="error-text">{openErr}</p>}

        <div ref={resultsRef} className="ev-results" aria-live="polite">
          {run ? (
            <RunHeader run={run} onSaved={refreshCurrent} />
          ) : (
            <div className="ev-runhead">
              <div className="ev-kpis">
                <Metric label="Evaluated images" value="—" sub="No run selected" />
                <Metric label="Detected objects" value="—" />
                <Metric label="Classification accuracy" value="—" sub="Needs a labelled run" />
                <Metric label="Avg inference time" value="—" />
              </div>
            </div>
          )}
          <div className="ev-grid">
            <ClassificationSection run={run} />
            <BinarySection run={run} />
            <DetectionSection run={run} />
            <InferenceSection run={run} />
            <PerImageTable run={run} />
          </div>
        </div>

        <div className="ev-grid">
          <HistorySection runs={history} currentId={run?.run_id} onOpen={open} onChanged={loadHistory} />
          <HistoricalSection items={s?.historical} />
        </div>

        <footer className="ev-foot muted small">
          Metrics describe the evaluated dataset only; they are not a guarantee of performance on other images. Station panels elsewhere on the site are
          educational and contain no measured results.
        </footer>
      </main>
    </div>
  );
}
