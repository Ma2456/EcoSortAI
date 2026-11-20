import { useEffect, useRef, useState } from 'react';

// Plotly is large, so it is loaded only when the evaluation page draws a chart.
let plotlyPromise = null;
const loadPlotly = () => (plotlyPromise ||= import('plotly.js-dist-min').then((m) => m.default || m));

// Chart tokens (dark surface of the site). Series slots validated with the
// dataviz palette validator against the panel surface #0f261a.
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#e66767', '#008300'];
export const INK = { primary: '#f3efe6', secondary: 'rgba(243,239,230,0.78)', muted: 'rgba(243,239,230,0.56)' };
export const GRID = 'rgba(243,239,230,0.10)';
// Sequential single-hue ramp (magnitude): panel surface → sage.
export const SEQ = [
  [0, '#12301f'],
  [0.25, '#24503a'],
  [0.5, '#3f7656'],
  [0.75, '#76a57a'],
  [1, '#c9e0b8'],
];

export const axis = (o = {}) => ({
  gridcolor: GRID,
  zeroline: false,
  linecolor: 'rgba(243,239,230,0.22)',
  tickfont: { color: INK.secondary, size: 11 },
  automargin: true,
  ...o,
  title: { font: { color: INK.muted, size: 11 }, ...(typeof o.title === 'string' ? { text: o.title } : o.title || {}) },
});

const BASE_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: "'Geist', ui-sans-serif, system-ui, sans-serif", color: INK.secondary, size: 12 },
  margin: { l: 56, r: 16, t: 40, b: 48 },
  hoverlabel: { bgcolor: '#0b1d14', bordercolor: 'rgba(243,239,230,0.22)', font: { color: INK.primary, size: 12 } },
  legend: { orientation: 'h', x: 0, y: 1.12, font: { color: INK.secondary, size: 11 } },
  bargap: 0.28,
  bargroupgap: 0.12,
};

export default function Plot({ data, layout = {}, height = 320, label, className = '' }) {
  const ref = useRef(null);
  const plotly = useRef(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    loadPlotly()
      .then((P) => {
        if (!alive || !ref.current) return;
        plotly.current = P;
        P.react(
          ref.current,
          data,
          { ...BASE_LAYOUT, ...layout, height, xaxis: axis(layout.xaxis), yaxis: axis(layout.yaxis) },
          { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'], toImageButtonOptions: { format: 'png', scale: 2 } }
        );
      })
      .catch((e) => alive && setError(e?.message || 'Chart library failed to load.'));
    return () => {
      alive = false;
    };
  }, [data, layout, height]);

  useEffect(() => () => ref.current && plotly.current?.purge(ref.current), []);

  if (error) return <p className="ev-empty">Chart unavailable: {error}</p>;
  return <div ref={ref} className={`ev-plot ${className}`} style={{ minHeight: height }} role="img" aria-label={label} />;
}
