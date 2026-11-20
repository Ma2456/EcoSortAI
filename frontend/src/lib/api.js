/**
 * EcoSortAI API client.
 * Base URL: VITE_API_BASE (e.g. https://api.example.com) or same-origin /api
 * (proxied to FastAPI on :8000 by vite.config.js during development).
 */
// Opened straight from disk (file://)? Talk to the local FastAPI server directly.
export const BASE = (
  import.meta.env.VITE_API_BASE ||
  (typeof location !== 'undefined' && location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '')
).replace(/\/$/, '');

export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
export const ACCEPTED_EXT = ['.jpg', '.jpeg', '.png', '.webp'];
export const DEFAULT_MAX_MB = 10;

export class ApiError extends Error {
  constructor(message, { status = 0, code = 'error', detail } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

export async function request(path, { method = 'GET', body, signal, timeoutMs = 15000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException('timeout', 'TimeoutError')), timeoutMs);
  const onAbort = () => ctrl.abort(signal.reason);
  signal?.addEventListener('abort', onAbort);
  try {
    const res = await fetch(`${BASE}${path}`, { method, body, signal: ctrl.signal });
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      /* non-JSON (e.g. proxy error page) */
    }
    if (!res.ok) {
      const d = data?.detail;
      const msg = (typeof d === 'object' ? d?.message : d) || data?.message || `Request failed (${res.status})`;
      throw new ApiError(msg, { status: res.status, code: (typeof d === 'object' && d?.code) || 'http_error', detail: data });
    }
    return data;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (signal?.aborted) throw new ApiError('Request cancelled', { code: 'cancelled' });
    if (ctrl.signal.aborted) throw new ApiError('The server took too long to respond.', { code: 'timeout' });
    throw new ApiError('Could not reach the EcoSortAI backend.', { code: 'offline' });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

export function getHealth(signal) {
  return request('/api/health', { signal, timeoutMs: 5000 });
}

export function analyzeImage(file, { explain = ['gradcam', 'lime', 'shap'], signal } = {}) {
  const fd = new FormData();
  fd.append('file', file, file.name || 'upload.jpg');
  fd.append('explain', explain.join(','));
  return request('/api/analyze', { method: 'POST', body: fd, signal, timeoutMs: 180000 });
}

export function validateFile(file, maxMb = DEFAULT_MAX_MB) {
  if (!file) return 'No file selected.';
  const name = (file.name || '').toLowerCase();
  const extOk = ACCEPTED_EXT.some((e) => name.endsWith(e));
  const typeOk = ACCEPTED_TYPES.includes(file.type);
  if (!typeOk && !extOk) return 'Unsupported file type. Please choose a JPG, JPEG, PNG or WebP image.';
  if (file.size === 0) return 'This file is empty.';
  if (file.size > maxMb * 1024 * 1024) return `This image is ${(file.size / 1048576).toFixed(1)} MB. The limit is ${maxMb} MB.`;
  return null;
}
