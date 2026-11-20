import { useMemo, useState } from 'react';
import Plot, { INK, SEQ, SERIES } from './Plot.jsx';
import { exportUrl } from '../../lib/evalApi.js';

// ------------------------------------------------------------------ formatting
export const pct = (v, d = 1) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`);
export const num = (v, d = 3) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));
export const ms = (v) => (v === null || v === undefined ? '—' : v >= 100 ? `${v.toFixed(0)} ms` : `${v.toFixed(1)} ms`);
export const fmtDate = (iso) => {
  try {
    return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  } catch {
    return iso;
  }
};

export function Info({ text }) {
  return (
    <span className="ev-info" tabIndex={0} aria-label={text} data-tip={text}>
      i
    </span>
  );
}

export function Metric({ label, value, sub, tip, tone }) {
  return (
    <div className={`ev-metric ${tone ? `ev-metric--${tone}` : ''}`}>
      <p className="ev-metric__label">
        {label}
        {tip && <Info text={tip} />}
      </p>
      <p className="ev-metric__value mono">{value}</p>
      {sub && <p className="ev-metric__sub">{sub}</p>}
    </div>
  );
}

export function Card({ title, eyebrow, actions, children, id, wide }) {
  return (
    <section className={`ev-card ${wide ? 'ev-card--wide' : ''}`} id={id} aria-labelledby={id ? `${id}-t` : undefined}>
      <header className="ev-card__head">
        <div>
          {eyebrow && <p className="eyebrow">{eyebrow}</p>}
          <h3 id={id ? `${id}-t` : undefined} className="ev-card__title">
            {title}
          </h3>
        </div>
        {actions && <div className="ev-card__actions">{actions}</div>}
      </header>
      {children}
    </section>
  );
}

export function Empty({ children }) {
  return <p className="ev-empty">{children}</p>;
}

export function CsvLink({ run, table, label = 'CSV' }) {
  return (
    <a className="btn btn--ghost btn--sm" href={exportUrl(run.run_id, 'csv', table)} download>
      {label}
    </a>
  );
}

// ------------------------------------------------------------------ classification
function Heatmap({ matrix, labels, colLabels, mode, height = 420, title }) {
  const { z, text } = useMemo(() => {
    const rows = matrix.map((r) => r.slice());
    let zz = rows;
    if (mode === 'row') zz = rows.map((r) => {
      const s = r.reduce((a, b) => a + b, 0);
      return r.map((v) => (s ? v / s : 0));
    });
    if (mode === 'col') {
      const cs = rows[0].map((_, j) => rows.reduce((a, r) => a + r[j], 0));
      zz = rows.map((r) => r.map((v, j) => (cs[j] ? v / cs[j] : 0)));
    }
    const tt = zz.map((r, i) => r.map((v, j) => (mode === 'count' ? String(rows[i][j]) : `${(v * 100).toFixed(0)}%`)));
    return { z: zz, text: tt };
  }, [matrix, mode]);
  const max = Math.max(1e-9, ...z.flat());
  const data = useMemo(
    () => [
      {
        type: 'heatmap',
        z,
        x: colLabels || labels,
        y: labels,
        text,
        texttemplate: '%{text}',
        textfont: { size: 12 },
        colorscale: SEQ,
        zmin: 0,
        zmax: mode === 'count' ? max : 1,
        xgap: 2,
        ygap: 2,
        hovertemplate:
          mode === 'count'
            ? 'True: %{y}<br>Predicted: %{x}<br>Images: %{z}<extra></extra>'
            : `True: %{y}<br>Predicted: %{x}<br>${mode === 'row' ? 'Share of true class' : 'Share of predictions'}: %{text}<extra></extra>`,
        colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: INK.muted, size: 10 }, tickformat: mode === 'count' ? 'd' : '.0%' },
      },
    ],
    [z, text, labels, colLabels, mode, max]
  );
  const layout = useMemo(
    () => ({
      margin: { l: 120, r: 20, t: 12, b: 90 },
      xaxis: { title: 'Predicted class', side: 'bottom', tickangle: -30, gridcolor: 'rgba(0,0,0,0)' },
      yaxis: { title: 'True class', autorange: 'reversed', gridcolor: 'rgba(0,0,0,0)' },
    }),
    []
  );
  return <Plot data={data} layout={layout} height={height} label={title} />;
}

export function ClassificationSection({ run }) {
  const c = run?.classification;
  const [mode, setMode] = useState('count');
  const bars = useMemo(() => {
    if (!c) return null;
    const rows = c.per_class.filter((r) => r.support > 0 || r.predicted > 0);
    const x = rows.map((r) => r.label);
    return [
      { type: 'bar', name: 'Precision', x, y: rows.map((r) => r.precision), marker: { color: SERIES[0] }, hovertemplate: '%{x}<br>Precision %{y:.3f}<extra></extra>' },
      { type: 'bar', name: 'Recall', x, y: rows.map((r) => r.recall), marker: { color: SERIES[1] }, hovertemplate: '%{x}<br>Recall %{y:.3f}<extra></extra>' },
      { type: 'bar', name: 'F1-score', x, y: rows.map((r) => r.f1), marker: { color: SERIES[2] }, hovertemplate: '%{x}<br>F1 %{y:.3f}<extra></extra>' },
    ];
  }, [c]);
  const barLayout = useMemo(() => ({ barmode: 'group', yaxis: { range: [0, 1.02], tickformat: '.0%', title: 'Score' }, xaxis: { tickangle: -20 } }), []);

  if (!c) {
    return (
      <Card title="Classification performance" eyebrow="Seven classes" id="ev-cls" wide>
        <Empty>
          {run && run.task !== 'classification' && run.task !== 'pipeline'
            ? 'This run did not evaluate the seven-class classifier. Run a classification or pipeline evaluation on a labelled dataset.'
            : run
              ? 'No classification metrics: the dataset had no usable ground-truth labels. Accuracy, precision, recall and F1 require labelled images.'
              : 'Run an evaluation, or open a saved run from the history below.'}
        </Empty>
      </Card>
    );
  }
  const names = c.label_names;
  const extra = c.n_abstained > 0;
  const matrix = extra ? c.confusion_matrix.map((r, i) => [...r, c.abstained_per_class[i]]) : c.confusion_matrix;
  return (
    <>
      <Card
        title="Classification performance"
        eyebrow={c.mode === 'pipeline' ? 'End-to-end hybrid pipeline' : 'Seven-class MobileNetV3'}
        id="ev-cls"
        wide
        actions={<CsvLink run={run} table="classification_report" label="Report CSV" />}
      >
        <p className="ev-note">{c.prediction_rule}</p>
        <div className="ev-metrics">
          <Metric label="Accuracy" value={pct(c.accuracy)} sub={`${c.n_correct} / ${c.n_samples} images`} tip="Correct predictions divided by all labelled images (images with no prediction count as wrong)." />
          <Metric label="Macro precision" value={num(c.macro_avg.precision)} tip="Unweighted mean over classes present in the ground truth." />
          <Metric label="Macro recall" value={num(c.macro_avg.recall)} tip="Unweighted mean over classes present in the ground truth." />
          <Metric label="Macro F1" value={num(c.macro_avg.f1)} tip="Unweighted mean of per-class F1 — every class counts equally." />
          <Metric label="Weighted F1" value={num(c.weighted_avg.f1)} sub={`P ${num(c.weighted_avg.precision)} · R ${num(c.weighted_avg.recall)}`} tip="Per-class F1 weighted by the number of true images in each class." />
          <Metric
            label="Coverage"
            value={pct(c.coverage)}
            sub={c.n_abstained ? `${c.n_abstained} with no prediction` : `${c.n_below_uncertainty_threshold} below ${pct(c.uncertainty_threshold, 0)} confidence`}
            tip="Share of images that received a category. Low-confidence predictions still count here; the website would show them as 'Uncertain'."
          />
        </div>
      </Card>

      <Card
        title="Confusion matrix"
        eyebrow="Rows = ground truth · columns = prediction"
        id="ev-cm"
        actions={
          <>
            <div className="ev-seg" role="radiogroup" aria-label="Normalisation">
              {[
                ['count', 'Counts'],
                ['row', 'Row %'],
                ['col', 'Column %'],
              ].map(([k, l]) => (
                <button key={k} type="button" role="radio" aria-checked={mode === k} className={mode === k ? 'is-on' : ''} onClick={() => setMode(k)}>
                  {l}
                </button>
              ))}
            </div>
            <CsvLink run={run} table="confusion_matrix" />
          </>
        }
      >
        <Heatmap matrix={matrix} labels={names} colLabels={extra ? [...names, 'No prediction'] : names} mode={mode} title="Seven-class confusion matrix" />
        <p className="ev-note small">
          Row % normalises each true class (the diagonal is recall); Column % normalises each predicted class (the diagonal is precision).
          {extra && ' “No prediction” = YOLO found nothing or the binary gate rejected every crop.'}
        </p>
      </Card>

      <Card title="Per-class report" eyebrow="Precision · recall · F1 · support" id="ev-report">
        <div className="ev-table-wrap">
          <table className="ev-table">
            <thead>
              <tr>
                <th scope="col">Class</th>
                <th scope="col">Precision</th>
                <th scope="col">Recall</th>
                <th scope="col">F1-score</th>
                <th scope="col">Support</th>
              </tr>
            </thead>
            <tbody>
              {c.per_class.map((r) => (
                <tr key={r.category} className={r.support === 0 ? 'is-dim' : ''}>
                  <th scope="row">{r.label}</th>
                  <td title={!r.precision_defined && r.precision !== null ? 'Never predicted: precision undefined, shown as 0' : undefined}>{num(r.precision)}</td>
                  <td>{num(r.recall)}</td>
                  <td>{num(r.f1)}</td>
                  <td>{r.support}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th scope="row">Accuracy</th>
                <td />
                <td />
                <td>{num(c.accuracy)}</td>
                <td>{c.n_samples}</td>
              </tr>
              <tr>
                <th scope="row">Macro avg</th>
                <td>{num(c.macro_avg.precision)}</td>
                <td>{num(c.macro_avg.recall)}</td>
                <td>{num(c.macro_avg.f1)}</td>
                <td>{c.n_samples}</td>
              </tr>
              <tr>
                <th scope="row">Weighted avg</th>
                <td>{num(c.weighted_avg.precision)}</td>
                <td>{num(c.weighted_avg.recall)}</td>
                <td>{num(c.weighted_avg.f1)}</td>
                <td>{c.n_samples}</td>
              </tr>
            </tfoot>
          </table>
        </div>
        {c.per_class.some((r) => r.support === 0) && <p className="ev-note small">Dimmed rows have no test images; they are excluded from the macro and weighted averages.</p>}
        <Plot data={bars} layout={barLayout} height={280} label="Per-class precision, recall and F1" />
      </Card>
    </>
  );
}

// ------------------------------------------------------------------ binary
export function BinarySection({ run }) {
  const b = run?.binary;
  const u = run?.binary_unsupervised;
  const hist = useMemo(() => {
    if (!b) return null;
    const e = b.probability_histogram.edges;
    const mids = e.slice(0, -1).map((v, i) => (v + e[i + 1]) / 2);
    return [
      { type: 'bar', name: 'True waste', x: mids, y: b.probability_histogram.waste, marker: { color: SERIES[2] }, hovertemplate: 'P(waste) ≈ %{x:.2f}<br>%{y} waste images<extra></extra>' },
      { type: 'bar', name: 'True non-waste', x: mids, y: b.probability_histogram.non_waste, marker: { color: SERIES[1] }, hovertemplate: 'P(waste) ≈ %{x:.2f}<br>%{y} non-waste images<extra></extra>' },
    ];
  }, [b]);
  const histLayout = useMemo(
    () => ({
      barmode: 'group',
      xaxis: { title: 'Predicted P(waste)', range: [0, 1] },
      yaxis: { title: 'Images', rangemode: 'tozero' },
      shapes: b ? [{ type: 'line', x0: b.threshold, x1: b.threshold, y0: 0, y1: 1, yref: 'paper', line: { color: INK.secondary, width: 1.5, dash: 'dot' } }] : [],
      annotations: b ? [{ x: b.threshold, y: 0.96, yref: 'paper', text: ` threshold ${b.threshold}`, showarrow: false, xanchor: 'left', yanchor: 'top', font: { color: INK.secondary, size: 11 } }] : [],
    }),
    [b]
  );
  if (!b) {
    return (
      <Card title="Binary waste verification" eyebrow="Waste / non-waste gate" id="ev-bin">
        {u ? (
          <>
            <div className="ev-metrics">
              <Metric label="Gate acceptance (unlabelled)" value={pct(u.overall_acceptance_rate)} sub={`${u.accepted} / ${u.n_samples} images`} tip={u.note} />
            </div>
            <Empty>{u.note} Add waste/ and non_waste/ folders to measure accuracy.</Empty>
          </>
        ) : (
          <Empty>Not part of this run. Run a binary verification evaluation with waste/ and non_waste/ folders.</Empty>
        )}
      </Card>
    );
  }
  return (
    <Card title="Binary waste verification" eyebrow={`MobileNetV3 gate · P(waste) ≥ ${b.threshold}`} id="ev-bin" actions={<CsvLink run={run} table="binary_metrics" />}>
      <div className="ev-metrics">
        <Metric label="Binary accuracy" value={pct(b.accuracy)} sub={`${b.tp + b.tn} / ${b.n_samples} correct`} tip="(TP + TN) / all labelled images." />
        <Metric label="Precision" value={num(b.precision)} tip="Of the images accepted as waste, the share that truly are waste." />
        <Metric label="Recall" value={num(b.recall)} tip="Of the truly-waste images, the share accepted." />
        <Metric label="F1-score" value={num(b.f1)} tip="Harmonic mean of precision and recall (waste = positive class)." />
        <Metric label="Waste acceptance rate" value={pct(b.waste_acceptance_rate)} sub={`${b.tp} of ${b.n_waste} true waste`} tip="P(accepted | truly waste). Conditioned on the true class — not accuracy." />
        <Metric label="Non-waste rejection rate" value={pct(b.non_waste_rejection_rate)} sub={`${b.tn} of ${b.n_non_waste} true non-waste`} tip="P(rejected | truly non-waste), i.e. specificity. Conditioned on the true class — not accuracy." />
      </div>
      <div className="ev-split">
        <div>
          <p className="ev-sub">Confusion matrix</p>
          <Heatmap matrix={b.confusion_matrix} labels={['Non-waste', 'Waste']} mode="count" height={260} title="Binary confusion matrix" />
        </div>
        <div>
          <p className="ev-sub">Score distribution</p>
          <Plot data={hist} layout={histLayout} height={260} label="Distribution of P(waste) by true label" />
        </div>
      </div>
      <p className="ev-note small">
        Balanced accuracy {num(b.balanced_accuracy)} · ROC-AUC {num(b.roc_auc)} · overall share accepted {pct(b.overall_acceptance_rate)} (all images, regardless of label).{' '}
        {run.binary_input}
      </p>
    </Card>
  );
}

// ------------------------------------------------------------------ detection
export function DetectionSection({ run }) {
  const d = run?.detection;
  if (!d || !d.class_aware?.available) {
    return (
      <Card title="YOLOv8 detection performance" eyebrow="Bounding boxes · not classification" id="ev-det">
        <Empty>
          {run?.task === 'detection'
            ? 'No detection metrics: the dataset has no YOLO bounding-box annotations. mAP is never computed from classification labels or confidence.'
            : 'Not part of this run. Run an object-detection evaluation with images/ + labels/ (YOLO format).'}
        </Empty>
      </Card>
    );
  }
  const ca = d.class_aware;
  const ag = d.class_agnostic;
  return (
    <Card title="YOLOv8 detection performance" eyebrow="Bounding boxes · not classification" id="ev-det" actions={<CsvLink run={run} table="detection_metrics" />}>
      <div className="ev-metrics">
        <Metric label="Precision" value={num(ca.precision)} tip="Box precision at the confidence that maximises mean F1 (Ultralytics convention)." />
        <Metric label="Recall" value={num(ca.recall)} tip="Box recall at the same operating confidence." />
        <Metric label="mAP@0.50" value={num(ca.map50)} tip="Mean over classes of AP at IoU 0.50." />
        <Metric label="mAP@0.50:0.95" value={num(ca.map50_95)} tip="Mean AP over IoU thresholds 0.50, 0.55 … 0.95." />
        <Metric label="Class-agnostic mAP@0.50" value={num(ag.map50)} sub={`mAP@0.50:0.95 ${num(ag.map50_95)}`} tip="Boxes matched ignoring class — measures localisation only." />
        <Metric label="Boxes" value={`${d.n_ground_truth_boxes}`} sub={`${d.n_predicted_boxes} predicted · ${d.n_images} images`} />
      </div>
      <div className="ev-table-wrap">
        <table className="ev-table">
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Instances</th>
              <th scope="col">Precision</th>
              <th scope="col">Recall</th>
              <th scope="col">mAP@0.50</th>
              <th scope="col">mAP@0.50:0.95</th>
            </tr>
          </thead>
          <tbody>
            {ca.per_class.map((r) => (
              <tr key={r.class_id}>
                <th scope="row">{r.name}</th>
                <td>{r.instances}</td>
                <td>{num(r.precision)}</td>
                <td>{num(r.recall)}</td>
                <td>{num(r.map50)}</td>
                <td>{num(r.map50_95)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="ev-note small">
        {d.methodology} conf ≥ {d.confidence_threshold}, NMS IoU {d.nms_iou_threshold}, max {d.max_det} boxes, agnostic NMS. Implementation: {d.implementation?.ap}.
      </p>
    </Card>
  );
}

// ------------------------------------------------------------------ inference
const STAGE_ORDER = ['preprocessing', 'detection', 'binary_filter', 'classification', 'gradcam', 'recommendation'];
const STAGE_SHORT = {
  preprocessing: 'Preprocessing',
  detection: 'YOLO detection',
  binary_filter: 'Binary filter',
  classification: 'CNN classification',
  gradcam: 'Grad-CAM',
  recommendation: 'Recommendation',
};

export function InferenceSection({ run }) {
  const inf = run?.inference;
  const stages = inf?.stages || {};
  const keys = STAGE_ORDER.filter((k) => stages[k]);
  const stageData = useMemo(
    () => [
      {
        type: 'bar',
        orientation: 'h',
        y: keys.map((k) => STAGE_SHORT[k]),
        x: keys.map((k) => stages[k].mean),
        customdata: keys.map((k) => [stages[k].median, stages[k].min, stages[k].max, stages[k].p95]),
        marker: { color: SERIES[0] },
        hovertemplate: '%{y}<br>Mean %{x:.1f} ms<br>Median %{customdata[0]:.1f} · min %{customdata[1]:.1f} · max %{customdata[2]:.1f} · p95 %{customdata[3]:.1f} ms<extra></extra>',
        text: keys.map((k) => ms(stages[k].mean)),
        textposition: 'outside',
        cliponaxis: false,
        textfont: { color: INK.secondary, size: 11 },
      },
    ],
    [inf] // eslint-disable-line react-hooks/exhaustive-deps
  );
  const stageLayout = useMemo(() => ({ margin: { l: 130, r: 70, t: 12, b: 40 }, xaxis: { title: 'Mean time per image (ms)', rangemode: 'tozero' }, yaxis: { autorange: 'reversed' } }), []);
  const totals = useMemo(() => (run?.per_image || []).map((r) => r.total_ms).filter((v) => typeof v === 'number'), [run]);
  const histData = useMemo(
    () => [{ type: 'histogram', x: totals, nbinsx: Math.min(30, Math.max(5, Math.round(totals.length / 2))), marker: { color: SERIES[2] }, hovertemplate: '%{x} ms<br>%{y} images<extra></extra>' }],
    [totals]
  );
  const histLayout = useMemo(() => ({ margin: { l: 56, r: 16, t: 12, b: 48 }, xaxis: { title: 'End-to-end time per image (ms)' }, yaxis: { title: 'Images' } }), []);

  if (!inf || !inf.end_to_end) {
    return (
      <Card title="Inference performance" eyebrow="Measured latency" id="ev-inf" wide>
        <Empty>Run any evaluation to measure inference time.</Empty>
      </Card>
    );
  }
  const e = inf.end_to_end;
  const inc = inf.includes || {};
  return (
    <Card title="Inference performance" eyebrow={`Measured on ${inf.device} · batch 1 · ${inf.n_images_timed} images after warm-up`} id="ev-inf" wide actions={<CsvLink run={run} table="inference_timing" />}>
      <div className="ev-metrics">
        <Metric label="Average" value={ms(e.mean)} tip="Mean end-to-end wall time per image for the stages ticked below." />
        <Metric label="Median" value={ms(e.median)} />
        <Metric label="Minimum" value={ms(e.min)} />
        <Metric label="Maximum" value={ms(e.max)} sub={`p95 ${ms(e.p95)}`} />
        <Metric label="Throughput" value={`${num(inf.fps, 2)} FPS`} tip={inf.fps_definition} />
      </div>
      <ul className="ev-includes" aria-label="Stages included in the timing">
        {[
          ['preprocessing', 'Image preprocessing'],
          ['detection', 'YOLO detection'],
          ['binary_filter', 'Binary filtering'],
          ['classification', 'CNN classification'],
          ['gradcam', 'Grad-CAM generation'],
          ['recommendation', 'Disposal recommendation'],
          ['lime_shap', 'LIME / SHAP'],
          ['http_upload_and_json', 'HTTP upload & JSON'],
        ].map(([k, l]) => (
          <li key={k} className={inc[k] ? 'is-in' : 'is-out'}>
            <span aria-hidden="true">{inc[k] ? '✓' : '–'}</span> {l}
            <span className="sr-only">{inc[k] ? ' included' : ' not included'}</span>
          </li>
        ))}
      </ul>
      <div className="ev-split">
        <div>
          <p className="ev-sub">Stage-wise execution time</p>
          <Plot data={stageData} layout={stageLayout} height={Math.max(180, 44 * keys.length + 60)} label="Mean time per pipeline stage" />
        </div>
        <div>
          <p className="ev-sub">Latency distribution</p>
          {totals.length > 1 ? <Plot data={histData} layout={histLayout} height={Math.max(180, 44 * keys.length + 60)} label="Histogram of end-to-end time per image" /> : <Empty>Needs at least two images.</Empty>}
        </div>
      </div>
      <p className="ev-note small">
        Stage times are measured separately with a high-resolution timer around each stage (with GPU synchronisation on CUDA). Model-specific resizing and
        normalisation happen inside each model stage. {inf.fps_definition}
      </p>
    </Card>
  );
}

// ------------------------------------------------------------------ historical (checkpoint metadata)
export function HistoricalSection({ items }) {
  if (!items?.length) return null;
  return (
    <Card title="Historical metrics recorded in the checkpoints" eyebrow="Not measured by this website" id="ev-hist-ck" wide>
      <p className="ev-note">
        These values were written into the model files by the original training scripts. They come from the training run’s own validation split, which is
        not included here, so they cannot be reproduced or verified from this project and are shown separately from measured runs.
      </p>
      <div className="ev-table-wrap">
        <table className="ev-table ev-table--left">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Recorded metric(s)</th>
              <th scope="col">Source</th>
              <th scope="col">Dataset / split</th>
              <th scope="col">Samples</th>
              <th scope="col">Checkpoint</th>
            </tr>
          </thead>
          <tbody>
            {items.map((h) => (
              <tr key={h.model}>
                <th scope="row">{h.model}</th>
                <td className="mono">
                  {Object.entries(h.metrics)
                    .map(([k, v]) => `${k.replace(/_/g, ' ')} ${v === null || v === undefined ? '—' : Number(v).toFixed(4)}`)
                    .join(' · ')}
                </td>
                <td>{h.source}</td>
                <td>{h.split}</td>
                <td>{h.sample_count ?? 'not recorded'}</td>
                <td className="mono small">
                  {h.checkpoint?.file}
                  <br />
                  {h.checkpoint?.sha256}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
