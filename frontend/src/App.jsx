import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { ScrollToPlugin } from 'gsap/ScrollToPlugin';
import Journey from './components/Journey.jsx';
import TopBar from './components/TopBar.jsx';
import WasteLab from './components/WasteLab.jsx';
import { buildSegments, holdStartUnits, STATIONS } from './config/timeline.js';
import { useHashRoute, useModelStatus, useReducedMotion } from './lib/hooks.js';

// Loaded on demand so the scroll journey's bundle is unchanged.
const EvaluationPage = lazy(() => import('./components/evaluation/EvaluationPage.jsx'));

gsap.registerPlugin(ScrollTrigger, ScrollToPlugin);
ScrollTrigger.config({ ignoreMobileResize: true });

export default function App() {
  const route = useHashRoute();
  if (route === 'evaluation') {
    return (
      <Suspense fallback={<p className="ev-loading">Loading Performance Evaluation…</p>}>
        <EvaluationPage />
      </Suspense>
    );
  }
  return <JourneyApp />;
}

function JourneyApp() {
  const { segments, totalUnits } = useMemo(() => buildSegments(), []);
  const reducedMotion = useReducedMotion();
  const model = useModelStatus();
  const journeyRef = useRef(null);
  const labRef = useRef(null);
  const [journeyState, setJourneyState] = useState({ station: -1, phase: 'intro', panelOpen: false, segment: 'intro' });
  const [inLab, setInLab] = useState(false);

  // Stable viewport unit in px (ignores mobile URL-bar resizes).
  useEffect(() => {
    let lastW = window.innerWidth;
    const set = () => document.documentElement.style.setProperty('--unit', `${window.innerHeight}px`);
    set();
    const onResize = () => {
      if (window.innerWidth !== lastW) {
        lastW = window.innerWidth;
        set();
        ScrollTrigger.refresh();
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Track when the lab is on screen (for the top bar indicator).
  useEffect(() => {
    const st = ScrollTrigger.create({
      trigger: labRef.current,
      start: 'top 55%',
      end: 'bottom top',
      onToggle: (self) => setInLab(self.isActive),
    });
    // Recalculate trigger positions whenever page height changes (results, fonts…).
    let raf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => ScrollTrigger.refresh());
    });
    ro.observe(labRef.current);
    return () => {
      st.kill();
      ro.disconnect();
      cancelAnimationFrame(raf);
    };
  }, []);

  const scrollToY = useCallback(
    (y, duration = 1.6) => {
      gsap.killTweensOf(window);
      if (reducedMotion || duration === 0) {
        window.scrollTo({ top: y, behavior: 'auto' });
        return;
      }
      gsap.to(window, { scrollTo: { y, autoKill: true }, duration, ease: 'power2.inOut' });
    },
    [reducedMotion]
  );

  const unitPx = () => window.innerHeight;
  const journeyTop = () => (journeyRef.current ? journeyRef.current.getBoundingClientRect().top + window.scrollY : 0);

  const goToStation = useCallback(
    (i) => {
      const clamped = Math.max(0, Math.min(STATIONS.length - 1, i));
      const y = journeyTop() + holdStartUnits(segments, clamped) * unitPx();
      const dist = Math.abs(y - window.scrollY) / unitPx();
      scrollToY(y, Math.min(3.2, 0.9 + dist * 0.25));
    },
    [segments, scrollToY]
  );

  const goToLab = useCallback(() => {
    const y = labRef.current.getBoundingClientRect().top + window.scrollY;
    scrollToY(y, 1.4);
    setTimeout(() => labRef.current?.querySelector('[data-autofocus]')?.focus({ preventScroll: true }), reducedMotion ? 50 : 1500);
  }, [scrollToY, reducedMotion]);

  const restart = useCallback(() => {
    scrollToY(0, reducedMotion ? 0 : 1.8);
  }, [scrollToY, reducedMotion]);

  const enterPark = useCallback(() => {
    // Glide down the path to the first bin.
    const y = journeyTop() + holdStartUnits(segments, 0) * unitPx();
    scrollToY(y, 3.0);
  }, [segments, scrollToY]);

  return (
    <>
      <a className="skip-link" href="#waste-lab" onClick={(e) => { e.preventDefault(); goToLab(); }}>
        Skip to Waste Lab
      </a>
      <TopBar
        journeyState={journeyState}
        inLab={inLab}
        onPrev={() => goToStation(inLab ? STATIONS.length - 1 : journeyState.station < 0 ? 0 : journeyState.station - 1)}
        onNext={() => (journeyState.station >= STATIONS.length - 1 ? goToLab() : goToStation(journeyState.station + 1))}
        onStation={goToStation}
        onLab={goToLab}
        onRestart={restart}
        modelStatus={model.status}
      />
      <main>
        <Journey
          ref={journeyRef}
          segments={segments}
          totalUnits={totalUnits}
          reducedMotion={reducedMotion}
          onStateChange={setJourneyState}
          onEnter={enterPark}
          model={model}
        />
        <WasteLab ref={labRef} model={model} onRestart={restart} />
      </main>
    </>
  );
}
