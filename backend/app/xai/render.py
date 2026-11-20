"""Image helpers for explanation overlays (no matplotlib dependency)."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# Compact "inferno"-like colour ramp.
_STOPS = np.array(
    [
        [0.0, 0, 0, 4],
        [0.25, 87, 16, 110],
        [0.5, 188, 55, 84],
        [0.75, 249, 142, 9],
        [1.0, 252, 255, 164],
    ],
    dtype=np.float32,
)


def colormap(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    out = np.empty(x.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(x, _STOPS[:, 0], _STOPS[:, c + 1])
    return out


def heat_overlay(base: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> Image.Image:
    base = base.convert("RGB")
    h = Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)
    h = np.asarray(h, dtype=np.float32) / 255.0
    col = colormap(h)
    a = (alpha * (0.25 + 0.75 * h))[..., None]
    out = np.asarray(base, dtype=np.float32) * (1 - a) + col * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def segment_overlay(base: Image.Image, segments: np.ndarray, weights: dict[int, float], max_abs: float | None = None) -> Image.Image:
    """Tint superpixels green (supports) / red (opposes) proportionally to weight, and outline them."""
    from skimage.segmentation import find_boundaries

    arr = np.asarray(base.convert("RGB"), dtype=np.float32)
    m = max_abs or (max((abs(v) for v in weights.values()), default=0) or 1.0)
    tint = np.zeros_like(arr)
    alpha = np.zeros(arr.shape[:2], dtype=np.float32)
    for seg, w in weights.items():
        mask = segments == seg
        s = min(1.0, abs(w) / m)
        if s < 0.05:
            continue
        tint[mask] = (88, 214, 120) if w > 0 else (239, 96, 80)
        alpha[mask] = 0.15 + 0.5 * s
    out = arr * (1 - alpha[..., None]) + tint * alpha[..., None]
    b = find_boundaries(segments, mode="thin")
    out[b] = out[b] * 0.4 + 255 * 0.6
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def to_data_url(img: Image.Image, fmt: str = "PNG", size: int | None = 320) -> str:
    if size:
        img = img.copy()
        img.thumbnail((size, size))
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, "JPEG", quality=88)
        mime = "image/jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def region_name(cx: float, cy: float) -> str:
    """Normalised point → plain-language region of the object crop."""
    v = "upper" if cy < 0.36 else "lower" if cy > 0.64 else "middle"
    h = "left" if cx < 0.36 else "right" if cx > 0.64 else "centre"
    if v == "middle" and h == "centre":
        return "centre"
    if v == "middle":
        return f"{h} side"
    return f"{v}-{h}" if h != "centre" else f"{v} centre"
