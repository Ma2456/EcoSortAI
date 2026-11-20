import { forwardRef, useCallback, useEffect, useRef, useState } from 'react';
import { analyzeImage, validateFile, ACCEPTED_EXT, ACCEPTED_TYPES, DEFAULT_MAX_MB } from '../lib/api.js';
import { DEMO_RESPONSE, DEMO_IMAGE } from '../content/demoResponse.js';
import ResultView from './ResultView.jsx';
import { UploadIcon, RestartIcon, CategoryIcon } from './Icons.jsx';
import { STATIONS } from '../config/timeline.js';
import { CATEGORIES } from '../content/wasteCategories.js';

const STAGES = ['Detecting objects', 'Classifying each object', 'Computing explanations'];

const WasteLab = forwardRef(function WasteLab({ model, onRestart }, ref) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [phase, setPhase] = useState('empty'); // empty | ready | analyzing | result | error
  const [validation, setValidation] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [result, setResult] = useState(null);
  const [demo, setDemo] = useState(false);
  const [drag, setDrag] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const inputRef = useRef(null);
  const ctrlRef = useRef(null);
  const resultRef = useRef(null);

  const maxMb = model.health?.limits?.max_upload_mb || DEFAULT_MAX_MB;

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);
  useEffect(() => () => ctrlRef.current?.abort(), []);

  useEffect(() => {
    if (phase !== 'analyzing') return;
    const t0 = performance.now();
    const id = setInterval(() => setElapsed((performance.now() - t0) / 1000), 200);
    return () => clearInterval(id);
  }, [phase]);

  const accept = useCallback(
    (f) => {
      const err = validateFile(f, maxMb);
      if (err) {
        setValidation(err);
        return;
      }
      setValidation(null);
      setApiError(null);
      setResult(null);
      setDemo(false);
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setPhase('ready');
    },
    [maxMb]
  );

  const remove = () => {
    ctrlRef.current?.abort();
    setFile(null);
    setPreview(null);
    setResult(null);
    setApiError(null);
    setValidation(null);
    setDemo(false);
    setPhase('empty');
    if (inputRef.current) inputRef.current.value = '';
  };

  const analyze = async () => {
    if (!file) return;
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setApiError(null);
    setPhase('analyzing');
    setElapsed(0);
    try {
      const res = await analyzeImage(file, { signal: ctrl.signal });
      setResult(res);
      setDemo(false);
      setPhase('result');
      requestAnimationFrame(() => resultRef.current?.focus());
    } catch (err) {
      if (err.code === 'cancelled') return;
      setApiError(err);
      setPhase('error');
      model.refresh();
    }
  };

  const cancel = () => {
    ctrlRef.current?.abort();
    setPhase('ready');
  };

  const showDemo = () => {
    setFile(null);
    setPreview(null);
    setApiError(null);
    setValidation(null);
    setResult(DEMO_RESPONSE);
    setDemo(true);
    setPhase('result');
    requestAnimationFrame(() => resultRef.current?.focus());
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) accept(f);
  };

  const canAnalyze = file && phase !== 'analyzing' && model.status !== 'offline';

  return (
    <section ref={ref} id="waste-lab" className="lab" aria-labelledby="lab-title">
      <div className="lab__bg" aria-hidden="true" />
      <div className="lab__inner">
        <header className="lab__head">
          <p className="eyebrow">The AI Waste Lab</p>
          <h2 id="lab-title" className="lab__title">
            Your Turn to <em>Make a Difference.</em>
          </h2>
          <p className="lab__lede">Upload an image of waste and discover how AI identifies it.</p>
          <ModelBadge model={model} />
          <ul className="lab__cats" aria-label="Waste categories">
            {STATIONS.map((s, i) => (
              <li key={s.id} style={{ '--accent': CATEGORIES[s.id].accent }}>
                <span className="mono">{String(i + 1).padStart(2, '0')}</span>
                <CategoryIcon id={s.id} />
                {CATEGORIES[s.id].short}
              </li>
            ))}
          </ul>
        </header>

        <div className="lab__grid">
          <div className="lab__upload">
            {!preview ? (
              <div
                className={`dropzone ${drag ? 'is-drag' : ''} ${validation ? 'is-invalid' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={onDrop}
              >
                <UploadIcon />
                <p className="dropzone__title">Drag &amp; drop a photo of waste</p>
                <p className="dropzone__sub">
                  JPG, JPEG, PNG or WebP · up to {maxMb} MB
                </p>
                <button type="button" className="btn btn--primary" data-autofocus onClick={() => inputRef.current?.click()}>
                  Browse files
                </button>
              </div>
            ) : (
              <figure className={`preview ${phase === 'analyzing' ? 'is-scanning' : ''}`}>
                <img src={preview} alt={`Selected image: ${file?.name}`} />
                {phase === 'analyzing' && (
                  <div className="scan" aria-hidden="true">
                    <span className="scan__line" />
                    <span className="scan__grid" />
                  </div>
                )}
                <figcaption>
                  <span className="mono truncate">{file?.name}</span>
                  <span className="mono muted">{(file.size / 1048576).toFixed(2)} MB</span>
                </figcaption>
              </figure>
            )}
            <input
              ref={inputRef}
              type="file"
              className="sr-only"
              accept={[...ACCEPTED_TYPES, ...ACCEPTED_EXT].join(',')}
              onChange={(e) => e.target.files?.[0] && accept(e.target.files[0])}
              aria-label="Choose an image of waste"
              tabIndex={-1}
            />
            {validation && (
              <p className="error-text" role="alert">
                {validation}
              </p>
            )}

            <div className="lab__actions">
              {preview && phase !== 'analyzing' && (
                <>
                  <button type="button" className="btn btn--primary" onClick={analyze} disabled={!canAnalyze}>
                    {phase === 'result' ? 'Analyze again' : 'Analyze Waste'}
                  </button>
                  <button type="button" className="btn btn--ghost" onClick={() => inputRef.current?.click()}>
                    Replace image
                  </button>
                  <button type="button" className="btn btn--ghost" onClick={remove}>
                    Remove
                  </button>
                </>
              )}
              {phase === 'analyzing' && (
                <>
                  <div className="analyzing" role="status" aria-live="polite">
                    <span className="spinner" aria-hidden="true" />
                    <span>
                      Analyzing… <span className="mono">{elapsed.toFixed(1)}s</span>
                    </span>
                  </div>
                  <button type="button" className="btn btn--ghost" onClick={cancel}>
                    Cancel
                  </button>
                </>
              )}
            </div>
            {phase === 'analyzing' && (
              <ol className="pipeline" aria-label="Pipeline stages">
                {(model.health?.detector ? STAGES : ['Whole-image classification', STAGES[2]]).map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ol>
            )}

            {phase === 'error' && apiError && <ErrorBox error={apiError} onRetry={analyze} model={model} />}
            {model.status === 'offline' && preview && phase !== 'error' && (
              <p className="muted small">The backend is offline, so analysis is unavailable. Start the API and press retry below.</p>
            )}
          </div>

          <aside className="lab__side">
            <h3>What happens to your image</h3>
            <ol className="how">
              <li>
                <b>Detect</b> — if an object detector is configured, each waste item gets a bounding box. Otherwise the whole image is treated as one item.
              </li>
              <li>
                <b>Classify</b> — the trained classifier scores the seven categories. Below the confidence threshold the result is reported as <i>uncertain</i>, not guessed.
              </li>
              <li>
                <b>Explain</b> — Grad-CAM, LIME and SHAP are computed from the same model on your image.
              </li>
              <li>
                <b>Recommend</b> — segregation, recycling, handling and disposal guidance for the predicted category.
              </li>
            </ol>
            <p className="muted small">Images are processed in memory by your configured backend and are not stored.</p>
            <button type="button" className="link-btn" onClick={showDemo}>
              View a sample result (demo data)
            </button>
          </aside>
        </div>

        {phase === 'result' && result && (
          <div ref={resultRef} tabIndex={-1} className="lab__result" aria-live="polite">
            <ResultView result={result} imageUrl={demo ? DEMO_IMAGE : preview} demo={demo} />
            <div className="lab__again">
              <button type="button" className="btn btn--primary" onClick={() => { remove(); requestAnimationFrame(() => inputRef.current?.click()); }}>
                Analyze another image
              </button>
              <button type="button" className="btn btn--ghost" onClick={onRestart}>
                <RestartIcon /> Walk the park again
              </button>
            </div>
          </div>
        )}

        <footer className="lab__foot">
          <p>
            EcoSortAI — Explainable AI-Based Waste Detection, Classification &amp; Recommendation System.
          </p>
          <p className="muted small">Station panels contain educational content; only the Waste Lab shows model output.</p>
        </footer>
      </div>
    </section>
  );
});

function ModelBadge({ model }) {
  const h = model.health;
  if (model.status === 'checking') return <p className="badge badge--neutral">Checking AI backend…</p>;
  if (model.status === 'offline')
    return (
      <p className="badge badge--warn">
        <span className="status-dot status-dot--offline" /> AI backend offline
        <button type="button" className="link-btn" onClick={() => model.refresh()}>
          Retry
        </button>
      </p>
    );
  if (model.status === 'no-model')
    return (
      <p className="badge badge--warn">
        <span className="status-dot status-dot--no-model" /> Backend online · no trained model configured
      </p>
    );
  return (
    <p className="badge badge--ok">
      <span className="status-dot status-dot--ready" /> Live model: {h.model?.name}
      {h.model?.kind === 'zero-shot-baseline' && <span className="badge__note">baseline, not trained on waste data</span>}
    </p>
  );
}

function ErrorBox({ error, onRetry, model }) {
  let hint = null;
  if (error.code === 'offline') hint = 'Make sure the FastAPI server is running (see README → Backend).';
  if (error.code === 'model_not_loaded') hint = 'Place your trained checkpoint in backend/model/ and configure model_config.json, then restart the API.';
  if (error.code === 'timeout') hint = 'Explanations (LIME/SHAP) can be slow on CPU. Try again or reduce the sample counts in backend settings.';
  return (
    <div className="error-box" role="alert">
      <p className="error-box__title">{error.message}</p>
      {hint && <p className="muted small">{hint}</p>}
      <button type="button" className="btn btn--ghost btn--sm" onClick={onRetry} disabled={model.status === 'checking'}>
        Retry
      </button>
    </div>
  );
}

export default WasteLab;
