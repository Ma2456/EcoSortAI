import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { STATIONS, VIDEO } from '../config/timeline.js';
import { sample } from './scrollMap.js';
import { objectPositionFor, videoRectToBox, clamp } from './geometry.js';

gsap.registerPlugin(ScrollTrigger);

const FRAME = 1 / VIDEO.fps;

/**
 * Drives the sticky stage from scroll position.
 *
 *  ScrollTrigger  → target scroll units (no React, no layout reads)
 *  gsap.ticker    → smoothed video time, CSS custom properties, seek scheduling
 *  onState        → called ONLY when station / phase / panel visibility changes
 *
 * Seeks are never queued: a new seek is issued only after the previous one has
 * finished ("seeked"), always to the latest target. That keeps scrubbing
 * responsive in both directions without piling up decoder work.
 */
export function createVideoEngine({ video, stage, camera, journey, segments, totalUnits, reducedMotion, mode, onState, onProgress }) {
  const state = {
    targetU: 0,
    smoothT: 0,
    lastKey: '',
    seeking: false,
    seekStart: 0,
    seekSamples: [],
    mode, // 'scrub' | 'playback'
    w: stage.clientWidth,
    h: stage.clientHeight,
    destroyed: false,
  };

  // Snap points: the intro, the middle of every bin's hold, and the end of the park.
  // When the user stops scrolling, the camera settles on the nearest stop, so a
  // fast flick can never skip a station (e.g. the last one).
  const snapPoints = [0, ...segments.filter((s) => s.kind === 'hold').map((s) => (s.u0 + (s.u1 - s.u0) * 0.45) / totalUnits), 1];

  const st = ScrollTrigger.create({
    trigger: journey,
    start: 'top top',
    end: 'bottom bottom',
    snap: {
      snapTo: snapPoints,
      duration: { min: 0.35, max: 1.1 },
      delay: 0.18,
      ease: 'power2.inOut',
      inertia: false,
    },
    onUpdate: (self) => {
      state.targetU = self.progress * totalUnits;
      onProgress?.(self.progress);
    },
  });
  state.targetU = st.progress * totalUnits;

  const onSeeking = () => {
    state.seeking = true;
  };
  const onSeeked = () => {
    state.seeking = false;
    if (state.seekStart) {
      const dt = performance.now() - state.seekStart;
      state.seekStart = 0;
      // Auto-fallback: if decoding a seek is consistently slow (some mobile
      // browsers), switch to rate-controlled playback instead of scrubbing.
      if (state.mode === 'scrub' && state.seekSamples.length < 24) {
        state.seekSamples.push(dt);
        if (state.seekSamples.length === 24) {
          const sorted = [...state.seekSamples].sort((a, b) => a - b);
          const median = sorted[12];
          if (median > 160) state.mode = 'playback';
        }
      }
    }
  };
  video.addEventListener('seeking', onSeeking);
  video.addEventListener('seeked', onSeeked);

  const onResize = () => {
    state.w = stage.clientWidth;
    state.h = stage.clientHeight;
  };
  window.addEventListener('resize', onResize);

  function seekTo(t) {
    if (state.seeking) return;
    if (Math.abs(video.currentTime - t) < FRAME * 0.5) return;
    state.seekStart = performance.now();
    video.currentTime = t;
  }

  function playbackTowards(t) {
    const cur = video.currentTime;
    const diff = t - cur;
    if (diff > FRAME * 2) {
      video.playbackRate = clamp(diff * 2.5, 0.5, 4);
      if (video.paused) video.play().catch(() => {});
    } else {
      if (!video.paused) video.pause();
      if (diff < -FRAME * 2) seekTo(t); // video can't play backwards: jump
    }
  }

  function tick(_time, deltaMs) {
    if (state.destroyed) return;
    const s = sample(segments, state.targetU, { reducedMotion });

    // Frame-rate independent smoothing of the video time.
    const k = reducedMotion ? 1 : 1 - Math.pow(1 - 0.2, (deltaMs || 16.7) / 16.7);
    state.smoothT += (s.time - state.smoothT) * k;
    if (Math.abs(s.time - state.smoothT) < FRAME * 0.25) state.smoothT = s.time;
    const t = clamp(state.smoothT, 0, VIDEO.duration - FRAME);

    if (video.readyState >= 1) {
      if (state.mode === 'playback') playbackTowards(t);
      else seekTo(t);
    }

    // Horizontal pan on narrow (portrait) viewports to keep the bin in view.
    const px = objectPositionFor(s.fx, state.w, state.h);
    video.style.objectPosition = `${(px * 100).toFixed(2)}% 50%`;

    // Stage variables for CSS-driven overlays.
    const style = stage.style;
    style.setProperty('--intro', s.intro.toFixed(3));
    style.setProperty('--focus', s.focus.toFixed(3));
    style.setProperty('--handoff', s.handoff.toFixed(3));

    if (s.station >= 0) {
      const station = STATIONS[s.station];
      const r = videoRectToBox(station.bin, state.w, state.h, px);
      style.setProperty('--bin-x', `${r.x.toFixed(1)}px`);
      style.setProperty('--bin-y', `${r.y.toFixed(1)}px`);
      style.setProperty('--bin-w', `${r.w.toFixed(1)}px`);
      style.setProperty('--bin-h', `${r.h.toFixed(1)}px`);
      style.setProperty('--bin-cx', `${(r.x + r.w / 2).toFixed(1)}px`);
      style.setProperty('--bin-cy', `${(r.y + r.h * 0.45).toFixed(1)}px`);
      // Gentle push-in toward the bin while inspecting (not in reduced motion).
      const zoom = reducedMotion ? 1 : 1 + s.focus * 0.08;
      camera.style.transformOrigin = `${(r.x + r.w / 2).toFixed(1)}px ${(r.y + r.h / 2).toFixed(1)}px`;
      camera.style.transform = `scale(${zoom.toFixed(4)})`;
    } else {
      camera.style.transform = 'scale(1)';
    }

    const key = `${s.station}|${s.phase}|${s.panelOpen ? 1 : 0}|${s.seg.kind}`;
    if (key !== state.lastKey) {
      state.lastKey = key;
      onState({ station: s.station, phase: s.phase, panelOpen: s.panelOpen, segment: s.seg.kind });
    }
  }

  gsap.ticker.add(tick);

  return {
    get mode() {
      return state.mode;
    },
    setMode(m) {
      state.mode = m;
      if (m === 'scrub' && !video.paused) video.pause();
    },
    refresh() {
      onResize();
      ScrollTrigger.refresh();
    },
    destroy() {
      state.destroyed = true;
      gsap.ticker.remove(tick);
      st.kill();
      video.removeEventListener('seeking', onSeeking);
      video.removeEventListener('seeked', onSeeked);
      window.removeEventListener('resize', onResize);
    },
  };
}

/**
 * Downloads the whole video into memory (Blob URL) with progress so every frame
 * is local before scrubbing starts. Falls back to the network URL on failure.
 */
export async function loadVideoBlob(url, onProgress, signal) {
  try {
    const res = await fetch(url, { signal });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const total = Number(res.headers.get('content-length')) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) onProgress?.(received / total);
    }
    onProgress?.(1);
    const type = url.endsWith('.webm') ? 'video/webm' : 'video/mp4';
    return { url: URL.createObjectURL(new Blob(chunks, { type })), blob: true };
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    return { url, blob: false };
  }
}
