export function Wordmark({ className = '' }) {
  return (
    <span className={`wordmark ${className}`}>
      <svg viewBox="0 0 24 24" aria-hidden="true" className="wordmark__leaf">
        <path d="M5 19c0-8 5-13.5 14-14.5-.6 9-5.6 14.5-13.5 14.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M5 19l8-8" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <span>
        EcoSort<b>AI</b>
      </span>
    </span>
  );
}

export function ArrowDown() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path d="M12 5v14M6 13l6 6 6-6" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function Chevron({ dir = 'up' }) {
  const d = dir === 'up' ? 'M6 15l6-6 6 6' : 'M6 9l6 6 6-6';
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function RestartIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4h4" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
      <path d="M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const CAT_PATHS = {
  plastic: 'M10 3h4v3c0 1 3 2 3 6v8a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1v-8c0-4 3-5 3-6z',
  paper: 'M6 3h9l3 3v15H6zM9 9h6M9 13h6M9 17h4',
  metal: 'M6 5c0-1 3-2 6-2s6 1 6 2v14c0 1-3 2-6 2s-6-1-6-2zM6 5c0 1 3 2 6 2s6-1 6-2',
  organic: 'M12 21c-5 0-8-4-8-8 0-5 4-8 8-8s8 3 8 8c0 4-3 8-8 8zM12 5c0-2 1-3 3-3',
  ewaste: 'M8 3h8a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM11 18h2',
  glass: 'M10 2h4v4c0 1 3 2 3 6v9H7v-9c0-4 3-5 3-6zM7 14h10',
  textile: 'M8 3l-5 4 3 4 2-1v11h8V10l2 1 3-4-5-4c-1 2-2 3-4 3S9 5 8 3z',
  hazardous: 'M12 3l9 16H3zM12 9v5M12 17v.5',
};

export function CategoryIcon({ id }) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" className="cat-icon">
      <path d={CAT_PATHS[id]} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
