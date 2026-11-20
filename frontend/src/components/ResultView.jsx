import { useState } from 'react';

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function ResultView({ result, imageUrl, demo }) {
  const objects = result.objects || [];
  const [sel, setSel] = useState(0);
  const obj = objects[sel];
  const { width: W, height: H } = result.image || { width: 1, height: 1 };

  return (
    <article className={`result ${demo ? 'is-demo' : ''}`}>
      {demo && (
        <div className="demo-banner" role="note">
          <strong>DEMO DATA</strong> — a hand-written sample response that shows the layout. It is not a model prediction.
        </div>
      )}
      <header className="result__head">
        <div>
          <p className="eyebrow">{demo ? 'Sample result' : 'Analysis result'}</p>
          <h3 className="result__title">
            {result.summary?.uncertain ? 'No Object/Waste Detected' : result.summary?.primary_label}
          </h3>
          <p className="muted">
            {result.summary?.object_count} object{result.summary?.object_count === 1 ? '' : 's'} classified
            {result.summary?.candidate_count != null && <> · {result.summary.candidate_count} YOLO candidates</>}
            {result.summary?.rejected_count > 0 && <> · {result.summary.rejected_count} rejected by binary filter</>}
            {result.model && (
              <>
                {' '}
                · model <span className="mono">{result.model.name}</span>
                {result.model.detector ? (
                  <>
                    {' '}
                    + detector <span className="mono">{result.model.detector.name}</span>
                  </>
                ) : (
                  ' · no object detector (whole image)'
                )}
              </>
            )}
            {result.timing_ms?.total != null && <> · {(result.timing_ms.total / 1000).toFixed(1)}s</>}
          </p>
        </div>
      </header>

      {result.warnings?.length > 0 && (
        <ul className="warnings">
          {result.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      <div className="result__grid">
        <figure className="result__image">
          <div className="bbox-wrap">
            {imageUrl && <img src={imageUrl} alt="Analysed image" />}
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
              {objects.map((o, i) =>
                o.bbox ? (
                  <g key={o.id} className={`bbox ${i === sel ? 'is-sel' : ''} ${o.prediction.uncertain ? 'is-unc' : ''}`}>
                    <rect x={o.bbox[0]} y={o.bbox[1]} width={o.bbox[2] - o.bbox[0]} height={o.bbox[3] - o.bbox[1]} vectorEffect="non-scaling-stroke" />
                  </g>
                ) : null
              )}
            </svg>
            {objects.map((o, i) =>
              o.bbox ? (
                <button
                  key={o.id}
                  type="button"
                  className={`bbox-tag ${i === sel ? 'is-sel' : ''} ${o.bbox[1] / H < 0.08 ? 'is-inside' : ''}`}
                  style={{ left: `${(o.bbox[0] / W) * 100}%`, top: `${(o.bbox[1] / H) * 100}%` }}
                  onClick={() => setSel(i)}
                  aria-label={`Select object ${i + 1}`}
                >
                  {i + 1} · {o.prediction.uncertain ? '?' : o.prediction.category_label}
                </button>
              ) : null
            )}
          </div>
        </figure>

        <div className="result__detail">
          {objects.length > 1 && (
            <div className="obj-tabs" role="tablist" aria-label="Detected objects">
              {objects.map((o, i) => (
                <button key={o.id} role="tab" type="button" aria-selected={i === sel} className="obj-tabs__tab" onClick={() => setSel(i)}>
                  <span className="mono">{String(i + 1).padStart(2, '0')}</span> {o.prediction.uncertain ? 'Uncertain' : o.prediction.category_label}
                </button>
              ))}
            </div>
          )}
          {obj && <ObjectDetail o={obj} demo={demo} />}
        </div>
      </div>
    </article>
  );
}

function ObjectDetail({ o, demo }) {
  const p = o.prediction;
  return (
    <div className="obj">
      <div className="obj__pred">
        <div>
          <span className="stat__label">Predicted category</span>
          <p className={`obj__cat ${p.uncertain ? 'is-unc' : ''}`}>{p.uncertain ? 'Uncertain / unknown' : p.category_label}</p>
          {p.uncertain && (
            <p className="muted small">
              The top score ({pct(p.confidence)} for {p.top_k?.[0]?.category_label}) is below the confidence threshold, so no category is claimed.
            </p>
          )}
          {o.detector && (
            <p className="muted small">
              Detector: <span className="mono">{o.detector.label}</span> ({pct(o.detector.confidence)})
            </p>
          )}
          {o.binary_waste_probability != null && (
            <p className="muted small">
              Binary waste probability: <span className="mono">{pct(o.binary_waste_probability)}</span>
            </p>
          )}
        </div>
        <div className="obj__conf">
          <span className="stat__label">Model confidence</span>
          <span className="mono obj__conf-v">{pct(p.confidence)}</span>
        </div>
      </div>

      {p.top_k?.length > 0 && (
        <div className="topk" aria-label="Top predictions">
          {p.top_k.map((k) => (
            <div className="topk__row" key={k.raw_label}>
              <span>
                {k.category_label}
                {k.raw_label !== k.category && <span className="muted mono small"> ({k.raw_label})</span>}
              </span>
              <span className="topk__bar">
                <span style={{ transform: `scaleX(${k.confidence})` }} />
              </span>
              <span className="mono">{pct(k.confidence)}</span>
            </div>
          ))}
        </div>
      )}

      {o.explanation && (
        <section className="block">
          <h4 className="block__label">Why the model decided this</h4>
          <p>{o.explanation.summary}</p>
          {o.explanation.evidence?.length > 0 && (
            <ul className="ticks">
              {o.explanation.evidence.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          )}
          {o.explanation.limitations?.length > 0 && (
            <ul className="ticks ticks--warn">
              {o.explanation.limitations.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          )}
        </section>
      )}

      <section className="block">
        <h4 className="block__label">Explainability {demo ? '(not generated for demo data)' : '— generated by the model'}</h4>
        <div className="xai-real">
          <XaiCard title="Grad-CAM" d={o.xai?.gradcam} alt="Grad-CAM heatmap overlaid on the object" />
          <XaiCard title="LIME" d={o.xai?.lime} alt="LIME superpixels supporting the prediction" />
          <XaiCard title="SHAP" d={o.xai?.shap} alt="SHAP region contributions" contributions />
        </div>
      </section>

      {o.recommendation && (
        <section className="block">
          <h4 className="block__label">Recommended action{o.recommendation.bin ? ` · ${o.recommendation.bin}` : ''}</h4>
          <dl className="action-list">
            {[
              ['Segregation', o.recommendation.segregation],
              ['Recycling / composting', o.recommendation.recycling],
              ['Safe handling', o.recommendation.handling],
              ['Disposal destination', o.recommendation.destination],
            ]
              .filter(([, v]) => v)
              .map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
          </dl>
        </section>
      )}
    </div>
  );
}

function XaiCard({ title, d, alt, contributions }) {
  if (!d || !d.available) {
    return (
      <figure className="xai-card is-off">
        <div className="xai-card__empty">{d?.note || 'Not available'}</div>
        <figcaption>
          <strong>{title}</strong>
        </figcaption>
      </figure>
    );
  }
  return (
    <figure className="xai-card">
      {d.image && <img src={d.image} alt={alt} />}
      <figcaption>
        <strong>{title}</strong> {d.note}
        {contributions && d.contributions?.length > 0 && (
          <span className="xai-card__contrib">
            {d.contributions.slice(0, 4).map((c) => (
              <span key={c.region} className="mono">
                {c.region}: {c.value > 0 ? '+' : ''}
                {c.value.toFixed(3)}
              </span>
            ))}
          </span>
        )}
      </figcaption>
    </figure>
  );
}
