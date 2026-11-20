import { SCROLL, STATIONS } from '../config/timeline.js';
import { lerp, smoothstep, clamp } from './geometry.js';

/**
 * Pure function: scroll units → everything the stage needs for that position.
 * No DOM access, so it is trivially testable and cheap to call every frame.
 */
export function sample(segments, u, { reducedMotion = false } = {}) {
  let seg = segments[segments.length - 1];
  for (let i = 0; i < segments.length; i++) {
    if (u < segments[i].u1) {
      seg = segments[i];
      break;
    }
  }
  const local = clamp((u - seg.u0) / Math.max(1e-6, seg.u1 - seg.u0));
  // Ease camera moves in/out so the dolly settles into each stop.
  const eased = seg.kind === 'approach' ? easeOutCubic(local) : seg.kind === 'depart' ? easeInCubic(local) : local;

  let time = lerp(seg.t0, seg.t1, eased);
  let fx = lerp(seg.fx0, seg.fx1, eased);

  // Reduced motion: no camera travel; cut straight between inspection frames.
  if (reducedMotion) {
    if (seg.station >= 0) {
      time = STATIONS[seg.station].inspect;
      const b = STATIONS[seg.station].bin;
      fx = (b.x0 + b.x1) / 2;
    } else if (seg.kind === 'intro') {
      time = 0;
    } else {
      time = seg.t1;
    }
  }

  // Focus envelope (spotlight, reticle, push-in) and panel visibility.
  let focus = 0;
  let phase = seg.kind;
  let panelOpen = false;
  if (seg.kind === 'hold') {
    focus = smoothstep(0, SCROLL.panelIn, local) * (1 - smoothstep(SCROLL.panelOut, 1, local));
    panelOpen = local >= SCROLL.panelIn * 0.5 && local < SCROLL.panelOut + 0.04;
    phase = local < SCROLL.panelIn ? 'arrive' : local < SCROLL.panelOut ? 'inspect' : 'leave';
  } else if (seg.kind === 'approach') {
    focus = smoothstep(0.7, 1, local) * 0.25;
  } else if (seg.kind === 'depart') {
    focus = (1 - smoothstep(0, 0.4, local)) * 0.0;
  }

  const intro = seg.kind === 'intro' ? 1 - smoothstep(0.1, 1, local) : 0;
  const handoff = seg.kind === 'handoff' ? smoothstep(0, 0.85, local) : seg.kind === 'outro' ? smoothstep(0.7, 1, local) * 0.35 : 0;

  return {
    seg,
    local,
    time,
    fx,
    focus,
    intro,
    handoff,
    phase,
    panelOpen,
    station: seg.station,
  };
}

const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3) * 0.6 - (1 - t) * 0.4 * (1 - t);
const easeInCubic = (t) => t * t * 0.6 + t * 0.4;
