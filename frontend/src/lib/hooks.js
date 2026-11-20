import { useEffect, useState, useCallback } from 'react';
import { getHealth } from './api.js';

export function useReducedMotion() {
  const query = '(prefers-reduced-motion: reduce)';
  const [reduced, setReduced] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setReduced(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return reduced;
}

export function useMediaQuery(query) {
  const [match, setMatch] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setMatch(mq.matches);
    mq.addEventListener('change', on);
    on();
    return () => mq.removeEventListener('change', on);
  }, [query]);
  return match;
}

/**
 * Backend + model status.
 * status: 'checking' | 'ready' (model loaded) | 'no-model' (API up, no model) | 'offline'
 */
export function useModelStatus() {
  const [info, setInfo] = useState({ status: 'checking', health: null, error: null });
  const refresh = useCallback(async (signal) => {
    setInfo((p) => ({ ...p, status: 'checking' }));
    try {
      const health = await getHealth(signal);
      setInfo({ status: health?.model_loaded ? 'ready' : 'no-model', health, error: null });
    } catch (err) {
      if (err.code === 'cancelled') return;
      setInfo({ status: 'offline', health: null, error: err.message });
    }
  }, []);
  useEffect(() => {
    const ctrl = new AbortController();
    refresh(ctrl.signal);
    return () => ctrl.abort();
  }, [refresh]);
  return { ...info, refresh };
}

/** Minimal hash router: '#/evaluation' → 'evaluation', anything else → 'journey'. */
export function useHashRoute() {
  const read = () => (typeof location !== 'undefined' && location.hash.startsWith('#/evaluation') ? 'evaluation' : 'journey');
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}
