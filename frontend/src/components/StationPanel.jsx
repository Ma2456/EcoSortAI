import { useEffect, useId, useRef, useState } from 'react';
import { TAB_LABELS } from '../content/wasteCategories.js';
import { GradCamIllustration, LimeIllustration, ShapIllustration } from './XaiIllustrations.jsx';
import { analyzeImage } from '../lib/api.js';
import { CategoryIcon } from './Icons.jsx';
import { STATIONS } from '../config/timeline.js';

export default function StationPanel({ index, station, category: c, open, layout, reducedMotion, model, captureBin }) {
  const [tab, setTab] = useState('identify');
  const [mounted, setMounted] = useState(open);
  const tabRefs = useRef([]);
  const uid = useId();

  // Keep it mounted through the exit transition, then unmount.
  useEffect(() => {
    if (open) setMounted(true);
    else {
      const t = setTimeout(() => setMounted(false), reducedMotion ? 0 : 520);
      return () => clearTimeout(t);
    }
  }, [open, reducedMotion]);

  // Each visit starts on the first tab.
  useEffect(() => {
    if (!open) {
      const t = setTimeout(() => setTab('identify'), 600);
      return () => clearTimeout(t);
    }
  }, [open]);

  if (!mounted) return null;

  const onTabKey = (e, i) => {
    let n = null;
    if (e.key === 'ArrowRight') n = (i + 1) % TAB_LABELS.length;
    if (e.key === 'ArrowLeft') n = (i - 1 + TAB_LABELS.length) % TAB_LABELS.length;
    if (e.key === 'Home') n = 0;
    if (e.key === 'End') n = TAB_LABELS.length - 1;
    if (n !== null) {
      e.preventDefault();
      setTab(TAB_LABELS[n].id);
      tabRefs.current[n]?.focus();
    }
  };

  const num = String(index + 1).padStart(2, '0');

  return (
    <aside
      className={`panel panel--${layout}`}
      data-open={open}
      style={{ '--accent': c.accent }}
      aria-label={`Station ${num}: ${c.name}`}
      aria-hidden={!open}
    >
      <header className="panel__head">
        <div className="panel__index">
          <CategoryIcon id={station.id} />
          <span>Station {num} / {String(STATIONS.length).padStart(2, '0')}</span>
        </div>
        <h2 className="panel__title">{c.name}</h2>
        <p className="panel__stream">{c.stream}</p>
        <p className="panel__tagline">{c.tagline}</p>
      </header>

      <div className="tabs" role="tablist" aria-label={`${c.name} information`}>
        {TAB_LABELS.map((t, i) => (
          <button
            key={t.id}
            ref={(el) => (tabRefs.current[i] = el)}
            role="tab"
            type="button"
            id={`${uid}-tab-${t.id}`}
            aria-controls={`${uid}-panel-${t.id}`}
            aria-selected={tab === t.id}
            tabIndex={open && tab === t.id ? 0 : -1}
            className="tabs__tab"
            onClick={() => setTab(t.id)}
            onKeyDown={(e) => onTabKey(e, i)}
          >
            <span className="tabs__n">{String.fromCharCode(65 + i)}</span>
            {t.label}
          </button>
        ))}
      </div>

      <div className="panel__body" role="tabpanel" id={`${uid}-panel-${tab}`} aria-labelledby={`${uid}-tab-${tab}`} key={tab}>
        {tab === 'identify' && <Identify c={c} />}
        {tab === 'ai' && <HowAi c={c} />}
        {tab === 'xai' && <Xai c={c} station={station} index={index} model={model} captureBin={captureBin} open={open} />}
        {tab === 'action' && <Action c={c} />}
      </div>
    </aside>
  );
}

function Identify({ c }) {
  return (
    <div className="stagger">
      <Block label="Example objects">
        <ul className="chips">
          {c.identify.examples.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </Block>
      <Block label="Typical visual characteristics">
        <ul className="ticks">
          {c.identify.characteristics.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </Block>
      <Block label="Common subcategories">
        <p className="inline-list">{c.identify.subcategories.join(' · ')}</p>
      </Block>
    </div>
  );
}

function HowAi({ c }) {
  return (
    <div className="stagger">
      <div className="demo-flag" role="note">
        Illustrative example — not a live model result
      </div>
      <div className="result-row">
        <Stat label="Detection" value="1 object" />
        <Stat label="Predicted class" value={c.ai.predictedClass} mono />
        <Stat label="Example confidence" value={c.ai.exampleConfidence.toFixed(2)} mono meter={c.ai.exampleConfidence} />
      </div>
      <Block label="Visual evidence a model typically uses">
        <ul className="ticks">
          {c.ai.cues.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </Block>
      <Block label="How the classification works">
        <p>{c.ai.process}</p>
      </Block>
    </div>
  );
}

function Xai({ c, station, index, model, captureBin, open }) {
  const [live, setLive] = useState({ state: 'idle' });
  const ctrlRef = useRef(null);
  useEffect(() => () => ctrlRef.current?.abort(), []);
  useEffect(() => {
    if (!open) ctrlRef.current?.abort();
  }, [open]);

  const runLive = async () => {
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setLive({ state: 'loading' });
    try {
      const file = await captureBin(index);
      const res = await analyzeImage(file, { explain: ['gradcam'], signal: ctrl.signal });
      setLive({ state: 'done', res });
    } catch (err) {
      if (err.code === 'cancelled') return;
      setLive({ state: 'error', message: err.message });
    }
  };

  const obj = live.res?.objects?.[0];

  return (
    <div className="stagger">
      <div className="demo-flag" role="note">
        Illustrations explain how to read each method — they are not computed outputs
      </div>
      <div className="xai-grid">
        <figure>
          <GradCamIllustration id={station.id} />
          <figcaption>
            <strong>Grad-CAM</strong>
            {c.xai.gradcam}
          </figcaption>
        </figure>
        <figure>
          <LimeIllustration id={station.id} />
          <figcaption>
            <strong>LIME</strong>
            {c.xai.lime}
          </figcaption>
        </figure>
      </div>
      <figure className="shap-fig">
        <ShapIllustration items={c.xai.shap} />
        <figcaption>
          <strong>SHAP</strong> Feature contributions (illustrative): positive values push toward {c.short.toLowerCase()}, negative away.
        </figcaption>
      </figure>
      <Block label="In plain language">
        <p>{c.xai.plain}</p>
      </Block>
      <Block label="Limitations">
        <ul className="ticks ticks--warn">
          {c.xai.limitations.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </Block>

      {model.status === 'ready' && (
        <div className="live-box">
          <p className="live-box__label">Live model</p>
          {live.state === 'idle' && (
            <button type="button" className="btn btn--ghost btn--sm" onClick={runLive}>
              Analyse this video frame with {model.health?.model?.name || 'the model'}
            </button>
          )}
          {live.state === 'loading' && <p className="muted">Running inference on the bin crop…</p>}
          {live.state === 'error' && (
            <p className="error-text">
              {live.message}{' '}
              <button type="button" className="link-btn" onClick={runLive}>
                Retry
              </button>
            </p>
          )}
          {live.state === 'done' && obj && (
            <div className="live-box__result">
              <p>
                Real model output on this frame:{' '}
                <strong>{obj.prediction.uncertain ? 'Uncertain' : obj.prediction.category_label}</strong>{' '}
                <span className="mono">({(obj.prediction.confidence * 100).toFixed(1)}%)</span>
              </p>
              {obj.xai?.gradcam?.available && <img src={obj.xai.gradcam.image} alt={`Grad-CAM heatmap computed by the model on the ${c.short} bin frame`} />}
              <p className="muted small">Video frames are synthetic footage; results may differ from real photos.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Action({ c }) {
  const rows = [
    ['Segregation', c.action.segregation],
    ['Recycling / composting', c.action.recycling],
    ['Safe handling', c.action.handling],
    ['Disposal destination', c.action.destination],
  ];
  return (
    <dl className="action-list stagger">
      {rows.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Block({ label, children }) {
  return (
    <section className="block">
      <h3 className="block__label">{label}</h3>
      {children}
    </section>
  );
}

function Stat({ label, value, mono, meter }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className={`stat__value ${mono ? 'mono' : ''}`}>{value}</span>
      {meter != null && (
        <span className="meter" aria-hidden="true">
          <span style={{ transform: `scaleX(${meter})` }} />
        </span>
      )}
    </div>
  );
}
