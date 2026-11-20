import { forwardRef, useCallback, useEffect, useRef, useState } from 'react';
import { STATIONS, VIDEO } from '../config/timeline.js';
import { CATEGORIES } from '../content/wasteCategories.js';
import { createVideoEngine, loadVideoBlob } from '../lib/videoEngine.js';
import { useMediaQuery } from '../lib/hooks.js';
import { videoRectToBox, objectPositionFor } from '../lib/geometry.js';
import StationPanel from './StationPanel.jsx';
import { Wordmark, ArrowDown } from './Icons.jsx';

function pickSource() {
  const q = new URLSearchParams(window.location.search).get('video');
  if (q && VIDEO.sources[q]) return VIDEO.sources[q];
  const small = Math.min(window.innerWidth, window.innerHeight) < 600;
  const saveData = navigator.connection?.saveData;
  const size = small || saveData ? 'mobile' : 'desktop';
  const probe = document.createElement('video');
  const h264 = probe.canPlayType('video/mp4; codecs="avc1.640028"') || probe.canPlayType('video/mp4; codecs="avc1.42E01E"');
  if (!h264 && probe.canPlayType('video/webm; codecs="vp9"')) return VIDEO.webmFallback[size];
  return VIDEO.sources[size];
}

const Journey = forwardRef(function Journey({ segments, totalUnits, reducedMotion, onStateChange, onEnter, model }, journeyRef) {
  const stageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const engineRef = useRef(null);
  const unlockedRef = useRef(false);

  const [src, setSrc] = useState(null);
  const [loadProgress, setLoadProgress] = useState(0);
  const [videoReady, setVideoReady] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [jState, setJState] = useState({ station: -1, phase: 'intro', panelOpen: false, segment: 'intro' });
  const [playbackMode, setPlaybackMode] = useState(false);
  const isNarrow = useMediaQuery('(max-width: 760px)');

  // 1) Fetch the video fully before scrubbing (with real progress).
  useEffect(() => {
    const ctrl = new AbortController();
    let objectUrl = null;
    const url = pickSource();
    loadVideoBlob(url, setLoadProgress, ctrl.signal)
      .then((r) => {
        if (r.blob) objectUrl = r.url;
        setSrc(r.url);
      })
      .catch(() => {});
    return () => {
      ctrl.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, []);

  // 2) Start the scroll engine once the element exists.
  const handleState = useCallback(
    (s) => {
      setJState(s);
      onStateChange(s);
    },
    [onStateChange]
  );

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stageRef.current || !journeyRef.current) return;
    const forcePlayback = new URLSearchParams(window.location.search).get('mode') === 'playback';
    const engine = createVideoEngine({
      video,
      stage: stageRef.current,
      camera: cameraRef.current,
      journey: journeyRef.current,
      segments,
      totalUnits,
      reducedMotion,
      mode: forcePlayback ? 'playback' : 'scrub',
      onState: handleState,
      onProgress: (p) => document.documentElement.style.setProperty('--journey-progress', p.toFixed(4)),
    });
    engineRef.current = engine;
    const poll = setInterval(() => setPlaybackMode(engine.mode === 'playback'), 1500);
    return () => {
      clearInterval(poll);
      engine.destroy();
      engineRef.current = null;
    };
  }, [segments, totalUnits, reducedMotion, handleState, journeyRef]);

  useEffect(() => {
    if (videoReady) engineRef.current?.refresh();
  }, [videoReady]);

  // iOS/Safari: a muted play() inside a gesture lets later seeks render frames.
  const unlock = useCallback(() => {
    const v = videoRef.current;
    if (!v || unlockedRef.current) return;
    unlockedRef.current = true;
    const p = v.play();
    if (p && p.then) p.then(() => v.pause()).catch(() => {});
  }, []);

  useEffect(() => {
    const once = () => unlock();
    window.addEventListener('pointerdown', once, { once: true, passive: true });
    window.addEventListener('touchstart', once, { once: true, passive: true });
    window.addEventListener('keydown', once, { once: true });
    return () => {
      window.removeEventListener('pointerdown', once);
      window.removeEventListener('touchstart', once);
      window.removeEventListener('keydown', once);
    };
  }, [unlock]);

  /** Grab the bin region of the current frame as a JPEG (for live-model analysis). */
  const captureBin = useCallback(async (stationIndex) => {
    const v = videoRef.current;
    if (!v || v.readyState < 2) throw new Error('Video frame not ready');
    const b = STATIONS[stationIndex].bin;
    const pad = 0.035;
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    const x0 = Math.max(0, (b.x0 - pad) * vw);
    const y0 = Math.max(0, (b.y0 - pad) * vh);
    const x1 = Math.min(vw, (b.x1 + pad) * vw);
    const y1 = Math.min(vh, (b.y1 + pad) * vh);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(x1 - x0);
    canvas.height = Math.round(y1 - y0);
    canvas.getContext('2d').drawImage(v, x0, y0, x1 - x0, y1 - y0, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((res) => canvas.toBlob(res, 'image/jpeg', 0.92));
    return new File([blob], `station-${STATIONS[stationIndex].id}-frame.jpg`, { type: 'image/jpeg' });
  }, []);

  // Decide where the panel goes so it never covers the bin.
  const station = jState.station >= 0 ? STATIONS[jState.station] : null;
  let panelLayout = 'sheet';
  if (station && !isNarrow && stageRef.current) {
    const w = stageRef.current.clientWidth;
    const h = stageRef.current.clientHeight;
    const r = videoRectToBox(station.bin, w, h, objectPositionFor((station.bin.x0 + station.bin.x1) / 2, w, h));
    const free = station.side === 'left' ? w - (r.x + r.w) : r.x;
    panelLayout = free >= 400 ? (station.side === 'left' ? 'right' : 'left') : 'sheet';
  }

  const loading = !videoReady && !videoError;
  const cat = station ? CATEGORIES[station.id] : null;

  return (
    <section
      ref={journeyRef}
      className="journey"
      style={{ height: `calc(var(--unit, 100vh) * ${totalUnits + 1})` }}
      aria-label="The Eco-Park journey"
    >
      <div ref={stageRef} className={`stage ${reducedMotion ? 'is-reduced' : ''}`} data-segment={jState.segment}>
        <div ref={cameraRef} className="camera">
          <video
            ref={videoRef}
            className="stage__video"
            src={src || undefined}
            poster={VIDEO.poster}
            muted
            playsInline
            preload="auto"
            disablePictureInPicture
            aria-hidden="true"
            tabIndex={-1}
            onLoadedData={() => setVideoReady(true)}
            onError={() => src && setVideoError(true)}
          />
          {videoError && <img className="stage__video stage__fallback" src={VIDEO.poster} alt="" aria-hidden="true" />}
          <div className="spotlight" aria-hidden="true" />
          {station && (
            <div className="reticle" aria-hidden="true" data-open={jState.panelOpen} data-chips={station.bin.y0 < 0.4 ? 'side' : 'top'}>
              <span className="reticle__c tl" />
              <span className="reticle__c tr" />
              <span className="reticle__c bl" />
              <span className="reticle__c br" />
              <span className="reticle__tag">
                {String(jState.station + 1).padStart(2, '0')} · {cat.short}
              </span>
              <ul className="reticle__items">
                {cat.identify.examples.slice(0, 3).map((ex, i) => (
                  <li key={ex} style={{ '--i': i }}>
                    {ex}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="stage__shade" aria-hidden="true" />
        <div className="stage__handoff" aria-hidden="true" />

        {/* Scene 1 — intro */}
        <div className="intro" data-active={jState.segment === 'intro'} aria-hidden={jState.segment !== 'intro'}>
          <div className="intro__inner">
            <Wordmark className="intro__mark" />
            <p className="intro__eyebrow">Explainable AI for waste detection, classification &amp; disposal</p>
            <h1 className="intro__title">
              Every Waste <em>Has a Story.</em>
            </h1>
            <p className="intro__lede">
              Explore how AI identifies waste, explains its decisions, and helps us dispose of it responsibly.
            </p>
            <button
              type="button"
              className="btn btn--primary intro__cta"
              onClick={() => {
                unlock();
                onEnter();
              }}
              tabIndex={jState.segment === 'intro' ? 0 : -1}
            >
              Enter the Park
              <ArrowDown />
            </button>
            <p className="intro__hint" aria-hidden="true">
              or scroll to walk the path
            </p>
          </div>
        </div>

        {/* Scene 2/3 — station panel */}
        {STATIONS.map((s, i) => (
          <StationPanel
            key={s.id}
            index={i}
            station={s}
            category={CATEGORIES[s.id]}
            open={jState.station === i && jState.panelOpen}
            layout={jState.station === i ? panelLayout : 'sheet'}
            reducedMotion={reducedMotion}
            model={model}
            captureBin={captureBin}
          />
        ))}

        {/* End of the park caption */}
        <div className="outro" data-active={jState.segment === 'outro' || jState.segment === 'handoff'} aria-hidden="true">
          <p>End of the path</p>
          <span>The Waste Lab is just ahead</span>
        </div>

        {loading && (
          <div className="loader" role="status" aria-live="polite">
            <div className="loader__bar">
              <span style={{ transform: `scaleX(${Math.max(0.04, loadProgress)})` }} />
            </div>
            <p>Preparing the park · {Math.round(loadProgress * 100)}%</p>
          </div>
        )}
        {videoError && (
          <p className="stage__notice" role="status">
            The park video could not be loaded — showing a still frame. Station content still follows your scroll.
          </p>
        )}
        {playbackMode && !videoError && (
          <p className="stage__notice stage__notice--subtle" role="status">
            Optimised playback mode for this device
          </p>
        )}
      </div>
    </section>
  );
});

export default Journey;
