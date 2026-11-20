/**
 * Schematic illustrations of how to read Grad-CAM, LIME and SHAP outputs.
 * They are deliberately drawn as diagrams (outlines, flat shapes) so they
 * cannot be mistaken for real model output.
 */
import { useId } from 'react';

const SHAPES = {
  plastic: (
    <g>
      <path d="M52 18h16v10c0 4 8 8 8 18v52c0 5-4 8-8 8H52c-4 0-8-3-8-8V46c0-10 8-14 8-18z" />
      <rect x="51" y="12" width="18" height="7" rx="2" />
      <path d="M44 60h32M44 74h32" />
    </g>
  ),
  paper: (
    <g>
      <path d="M26 34l58-10 10 62-58 10z" />
      <path d="M36 44l40-7M38 54l40-7M40 64l40-7M42 74l26-4" />
      <path d="M20 88l74-12" />
    </g>
  ),
  metal: (
    <g>
      <ellipse cx="60" cy="26" rx="22" ry="7" />
      <path d="M38 26v62c0 4 10 7 22 7s22-3 22-7V26" />
      <path d="M38 34c0 4 10 7 22 7s22-3 22-7M38 80c0 4 10 7 22 7s22-3 22-7" />
      <ellipse cx="64" cy="25" rx="6" ry="2.5" />
    </g>
  ),
  organic: (
    <g>
      <path d="M22 86c4-20 16-30 30-30 8 0 14 6 18 12 4-10 14-16 24-12 6 3 8 12 6 20z" />
      <path d="M40 60c-6-10-2-20 8-24M74 58c2-10 10-16 18-14" />
      <circle cx="56" cy="74" r="7" />
      <path d="M30 80c8-4 14-4 20 0" />
    </g>
  ),
  ewaste: (
    <g>
      <rect x="30" y="20" width="36" height="64" rx="6" />
      <rect x="35" y="27" width="26" height="42" rx="2" />
      <path d="M66 70c14 0 14 18 26 18h6" />
      <rect x="92" y="83" width="10" height="10" rx="2" />
      <circle cx="48" cy="77" r="2.5" />
    </g>
  ),
  glass: (
    <g>
      <path d="M53 12h14v18c0 6 12 10 12 24v40c0 4-3 7-7 7H48c-4 0-7-3-7-7V54c0-14 12-18 12-24z" />
      <path d="M50 70c6-4 14-4 20 0" />
      <path d="M49 46v34" />
    </g>
  ),
  hazardous: (
    <g>
      <path d="M48 14h24v10l8 8v58c0 4-3 7-7 7H47c-4 0-7-3-7-7V32l8-8z" />
      <path d="M60 46l14 24H46z" />
      <path d="M60 54v8M60 66v1" />
    </g>
  ),
  textile: (
    <g>
      <path d="M28 40l16-14h32l16 14-10 12-8-6v46H46V46l-8 6z" />
      <path d="M50 26c2 6 6 9 10 9s8-3 10-9" />
      <path d="M46 64c8 3 20 3 28 0M46 78c8 3 20 3 28 0" />
    </g>
  ),
};

// Where the "important" evidence sits on each schematic (centre x, y, radius).
const HOT = {
  plastic: [[60, 28, 22], [60, 66, 16]],
  paper: [[58, 50, 26], [84, 82, 12]],
  metal: [[60, 27, 22], [66, 60, 12]],
  organic: [[50, 64, 22], [82, 60, 16]],
  ewaste: [[80, 82, 16], [48, 46, 18]],
  glass: [[60, 24, 18], [60, 70, 20]],
  textile: [[60, 34, 18], [60, 72, 20]],
  hazardous: [[60, 60, 20], [60, 20, 12]],
};

export function GradCamIllustration({ id }) {
  const uid = useId().replace(/:/g, '');
  return (
    <svg className="xai-svg" viewBox="0 0 120 110" role="img" aria-label="Diagram: Grad-CAM heat overlay on an object outline">
      <defs>
        <radialGradient id={`h${uid}`}>
          <stop offset="0" stopColor="#ffdd66" stopOpacity="0.95" />
          <stop offset="0.45" stopColor="#ff8a3d" stopOpacity="0.6" />
          <stop offset="1" stopColor="#ff3d3d" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="120" height="110" rx="8" className="xai-svg__bg" />
      <g className="xai-svg__shape">{SHAPES[id]}</g>
      {HOT[id].map(([x, y, r], i) => (
        <circle key={i} cx={x} cy={y} r={r} fill={`url(#h${uid})`} style={{ mixBlendMode: 'screen' }} />
      ))}
      <text x="6" y="104" className="xai-svg__tag">schematic</text>
    </svg>
  );
}

export function LimeIllustration({ id }) {
  // Fixed superpixel tiling; the ones overlapping the hot zones are "supporting".
  const cells = [];
  const cols = 6;
  const rows = 5;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * 20 + (r % 2 ? 4 : 0);
      const y = r * 22;
      const cx = x + 10;
      const cy = y + 11;
      const hot = HOT[id].some(([hx, hy, hr]) => Math.hypot(cx - hx, cy - hy) < hr * 0.9);
      const against = !hot && (r * 7 + c * 3 + id.length) % 11 === 0;
      cells.push(
        <path
          key={`${r}-${c}`}
          d={`M${x + 2} ${y + 3} l${16 + ((r + c) % 3)} ${-1 + (c % 2)} l${1 - (r % 2)} ${18 + (c % 3)} l${-17} ${1} z`}
          className={hot ? 'lime-cell lime-cell--pos' : against ? 'lime-cell lime-cell--neg' : 'lime-cell'}
        />
      );
    }
  }
  return (
    <svg className="xai-svg" viewBox="0 0 120 110" role="img" aria-label="Diagram: LIME superpixels supporting or opposing the prediction">
      <rect width="120" height="110" rx="8" className="xai-svg__bg" />
      <g className="xai-svg__shape xai-svg__shape--dim">{SHAPES[id]}</g>
      {cells}
      <text x="6" y="104" className="xai-svg__tag">schematic</text>
    </svg>
  );
}

export function ShapIllustration({ items }) {
  const max = Math.max(...items.map((i) => Math.abs(i.weight)));
  return (
    <div className="shap" role="img" aria-label={`Illustrative SHAP contributions: ${items.map((i) => `${i.feature} ${i.weight > 0 ? '+' : ''}${i.weight}`).join(', ')}`}>
      {items.map((it) => {
        const w = (Math.abs(it.weight) / max) * 50;
        return (
          <div className="shap__row" key={it.feature}>
            <span className="shap__label">{it.feature}</span>
            <span className="shap__track">
              <span className="shap__axis" />
              <span
                className={`shap__bar ${it.weight < 0 ? 'is-neg' : ''}`}
                style={it.weight >= 0 ? { left: '50%', width: `${w}%` } : { right: '50%', width: `${w}%` }}
              />
            </span>
            <span className="shap__val mono">
              {it.weight > 0 ? '+' : ''}
              {it.weight.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
