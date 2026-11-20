/**
 * EcoSortAI — video timeline configuration
 * ---------------------------------------------------------------------------
 * Source of truth: Camera_journey_through_eco-park_20261015162017.mp4
 * (8.0 s, 1280×720, 24 fps). Every timestamp below was chosen by sampling the
 * footage at 4 fps and then frame-by-frame around each bin.
 *
 * Observations from the footage:
 *  - The camera dollies forward along one straight path; it never stops, so the
 *    "pause" at each bin is a hold of the scroll-mapped video time on one frame.
 *  - Seven bins, in this order: plastic (blue, L) → paper (green, R) → metal
 *    (grey can, L) → organic (brown, R) → e-waste (charcoal, lid open, L) →
 *    hazardous (red, skull label, lid open, L) → glass (clear, R).
 *  - No lid opens on camera (the blue, e-waste and hazardous lids are already
 *    open), so the site uses a spotlight + reticle overlay instead of a fake lid.
 *  - The clip ends ~0.1 s after the glass bin leaves frame; the last frame hands
 *    over to the Waste Lab.
 *
 * Fields: inspect = pause frame (s); approachFrom / departUntil = movement
 * window; bin = bin box at the pause frame, normalised 0–1 (x0, y0, x1, y1);
 * side = side of the path.
 */

export const VIDEO = {
  duration: 8.0,
  fps: 24,
  aspect: 16 / 9,
  // Web encodes of the supplied MP4 (all-intra H.264, audio removed) so any scroll
  // position decodes a single frame. The untouched original is shipped too and can
  // be selected with ?video=original.
  sources: {
    desktop: '/media/park-720.mp4',
    mobile: '/media/park-480.mp4',
    original: '/media/original/Camera_journey_through_eco-park_20261015162017.mp4',
  },
  // VP9 fallbacks (same footage) for browsers that cannot decode H.264.
  webmFallback: {
    desktop: '/media/park-720.webm',
    mobile: '/media/park-480.webm',
  },
  poster: '/media/poster.jpg',
  endFrame: '/media/end.jpg',
};

export const STATIONS = [
  {
    id: 'plastic',
    approachFrom: 0.0,
    inspect: 1.8,
    departUntil: 2.3,
    side: 'left',
    bin: { x0: 0.14, y0: 0.48, x1: 0.3, y1: 0.89 },
    footageNote: 'Blue wheelie bin, lid propped open, PET bottles at the rim.',
  },
  {
    id: 'paper',
    approachFrom: 2.3,
    inspect: 2.8,
    departUntil: 3.3,
    side: 'right',
    bin: { x0: 0.69, y0: 0.48, x1: 0.86, y1: 0.95 },
    footageNote: 'Green wheelie bin topped with newspaper and cardboard.',
  },
  {
    id: 'metal',
    approachFrom: 3.3,
    inspect: 4.6,
    departUntil: 4.9,
    side: 'left',
    bin: { x0: 0.14, y0: 0.53, x1: 0.28, y1: 0.9 },
    footageNote: 'Galvanised grey bin full of aluminium cans.',
  },
  {
    id: 'organic',
    approachFrom: 4.9,
    inspect: 5.05,
    departUntil: 5.55,
    side: 'right',
    bin: { x0: 0.7, y0: 0.49, x1: 0.86, y1: 0.91 },
    footageNote: 'Brown wheelie bin heaped with fruit and vegetable scraps.',
  },
  {
    id: 'ewaste',
    approachFrom: 5.55,
    inspect: 6.0,
    departUntil: 6.5,
    side: 'left',
    bin: { x0: 0.15, y0: 0.3, x1: 0.3, y1: 0.84 },
    footageNote: 'Charcoal bin with its lid open, electronic scrap inside.',
  },
  {
    id: 'hazardous',
    approachFrom: 6.5,
    inspect: 6.95,
    departUntil: 7.25,
    side: 'left',
    bin: { x0: 0.08, y0: 0.28, x1: 0.29, y1: 0.88 },
    footageNote: 'Red wheelie bin with skull-and-crossbones labels, lid open, chemical containers inside.',
  },
  {
    id: 'glass',
    approachFrom: 7.25,
    inspect: 7.45,
    departUntil: 7.85,
    side: 'right',
    bin: { x0: 0.72, y0: 0.46, x1: 0.91, y1: 0.92 },
    footageNote: 'Transparent bin holding green and clear glass bottles.',
  },
];

export const END_TIME = 7.96;

/* -------------------------------------------------------------------------
 * Scroll map
 * Scroll is measured in "units" (1 unit = 1 viewport height). Each segment
 * maps a scroll range to a video-time range (or a hold on a single frame).
 * ----------------------------------------------------------------------- */
export const SCROLL = {
  introHold: 0.6, // scroll units over the first frame while the title fades out
  unitsPerSecond: 0.9, // scroll units per second of camera motion
  minMove: 0.5, // minimum units for any moving segment
  hold: 2.4, // units the camera holds at each bin (panel in → read → out)
  panelIn: 0.14, // fraction of the hold where the panel has fully appeared
  panelOut: 0.82, // fraction of the hold where the panel starts leaving
  outro: 0.8, // units from the last bin to the end of the park
  labHandOff: 0.9, // units holding the last frame while the lab rises
};

/**
 * Builds the ordered list of segments. Each segment:
 *  { kind: 'intro'|'approach'|'hold'|'depart'|'outro'|'handoff',
 *    station: index|-1, u0, u1, t0, t1, fx0, fx1 }
 * fx = horizontal focus point (0–1 of the video width) used to pan the frame
 * on viewports narrower than 16:9 so the active bin stays visible.
 */
export function buildSegments() {
  const segs = [];
  let u = 0;
  const push = (kind, station, du, t0, t1, fx0, fx1) => {
    segs.push({ kind, station, u0: u, u1: u + du, t0, t1, fx0, fx1 });
    u += du;
  };
  const move = (dt) => Math.max(SCROLL.minMove, dt * SCROLL.unitsPerSecond);
  const cx = (s) => (s.bin.x0 + s.bin.x1) / 2;

  push('intro', -1, SCROLL.introHold, 0, 0, 0.5, 0.5);
  STATIONS.forEach((s, i) => {
    const prevFx = i === 0 ? 0.5 : 0.5; // camera heads down the path between bins
    push('approach', i, move(s.inspect - s.approachFrom), s.approachFrom, s.inspect, prevFx, cx(s));
    push('hold', i, SCROLL.hold, s.inspect, s.inspect, cx(s), cx(s));
    push('depart', i, move(s.departUntil - s.inspect), s.inspect, s.departUntil, cx(s), 0.5);
  });
  const last = STATIONS[STATIONS.length - 1];
  push('outro', -1, SCROLL.outro, last.departUntil, END_TIME, 0.5, 0.5);
  push('handoff', -1, SCROLL.labHandOff, END_TIME, END_TIME, 0.5, 0.5);
  return { segments: segs, totalUnits: u };
}

/** Scroll-unit position where a station's hold begins (used for jump links). */
export function holdStartUnits(segments, stationIndex) {
  const seg = segments.find((s) => s.kind === 'hold' && s.station === stationIndex);
  return seg ? seg.u0 + (seg.u1 - seg.u0) * (SCROLL.panelIn + 0.02) : 0;
}
