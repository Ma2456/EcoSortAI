import { STATIONS } from '../config/timeline.js';
import { CATEGORIES } from '../content/wasteCategories.js';
import { Wordmark, Chevron, RestartIcon } from './Icons.jsx';

const TOTAL = String(STATIONS.length).padStart(2, '0');

export default function TopBar({ journeyState, inLab, onPrev, onNext, onStation, onLab, onRestart, modelStatus }) {
  const i = journeyState.station;
  const label = inLab
    ? 'Waste Lab'
    : i >= 0
      ? CATEGORIES[STATIONS[i].id].short
      : journeyState.segment === 'intro'
        ? 'Park entrance'
        : 'End of the path';

  return (
    <header className={`topbar ${inLab ? 'is-lab' : ''}`}>
      <div className="topbar__progress" aria-hidden="true">
        <span />
      </div>
      <button type="button" className="topbar__brand" onClick={onRestart} aria-label="EcoSortAI — restart the journey">
        <Wordmark />
      </button>

      <nav className="topbar__nav" aria-label="Journey">
        <div className="station-ind" aria-live="polite">
          <span className="station-ind__num mono">
            {inLab ? 'LAB' : i >= 0 ? `${String(i + 1).padStart(2, '0')} / ${TOTAL}` : `— / ${TOTAL}`}
          </span>
          <span className="station-ind__name">{label}</span>
        </div>
        <ol className="dots">
          {STATIONS.map((s, k) => (
            <li key={s.id}>
              <button
                type="button"
                className="dots__dot"
                aria-current={k === i && !inLab ? 'step' : undefined}
                aria-label={`Go to station ${k + 1}: ${CATEGORIES[s.id].name}`}
                onClick={() => onStation(k)}
                style={{ '--accent': CATEGORIES[s.id].accent }}
              />
            </li>
          ))}
        </ol>
        <div className="topbar__step">
          <button type="button" className="icon-btn" onClick={onPrev} aria-label="Previous station" disabled={inLab ? false : i <= 0}>
            <Chevron dir="up" />
          </button>
          <button type="button" className="icon-btn" onClick={onNext} aria-label={i >= STATIONS.length - 1 ? 'Go to the Waste Lab' : 'Next station'} disabled={inLab}>
            <Chevron dir="down" />
          </button>
        </div>
      </nav>

      <div className="topbar__actions">
        <a className="topbar__perf" href="#/evaluation" title="Measured model performance">
          Performance
        </a>
        <button type="button" className="link-btn topbar__lab" onClick={onLab}>
          <span className={`status-dot status-dot--${modelStatus}`} aria-hidden="true" />
          Skip to Waste Lab
        </button>
        <button type="button" className="icon-btn" onClick={onRestart} aria-label="Restart the experience" title="Restart">
          <RestartIcon />
        </button>
      </div>
    </header>
  );
}
