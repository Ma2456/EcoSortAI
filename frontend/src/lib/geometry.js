import { VIDEO } from '../config/timeline.js';

/**
 * Geometry for a video rendered with `object-fit: cover` inside a box of
 * (w, h) with a horizontal `object-position` of px (0–1) and vertical 50%.
 */
export function coverBox(w, h, px = 0.5, py = 0.5, aspect = VIDEO.aspect) {
  const boxAspect = w / h;
  let vw, vh;
  if (boxAspect > aspect) {
    vw = w;
    vh = w / aspect;
  } else {
    vh = h;
    vw = h * aspect;
  }
  const ox = (w - vw) * px; // <= 0
  const oy = (h - vh) * py;
  return { vw, vh, ox, oy };
}

/** Fraction of the video width visible in the box. */
export function visibleFraction(w, h, aspect = VIDEO.aspect) {
  const { vw } = coverBox(w, h, 0.5, 0.5, aspect);
  return Math.min(1, w / vw);
}

/**
 * object-position (0–1) that centres a normalised video x-coordinate fx in the
 * box, clamped so the video still covers the box.
 */
export function objectPositionFor(fx, w, h) {
  const f = visibleFraction(w, h);
  if (f >= 0.999) return 0.5;
  const left = fx - f / 2;
  return Math.min(1, Math.max(0, left / (1 - f)));
}

/** Normalised video rect → box pixel rect. */
export function videoRectToBox(rect, w, h, px = 0.5) {
  const { vw, vh, ox, oy } = coverBox(w, h, px);
  return {
    x: ox + rect.x0 * vw,
    y: oy + rect.y0 * vh,
    w: (rect.x1 - rect.x0) * vw,
    h: (rect.y1 - rect.y0) * vh,
  };
}

export const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
