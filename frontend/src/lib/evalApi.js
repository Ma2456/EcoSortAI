/**
 * Client for the Performance Evaluation API (/api/evaluation/*).
 * Uses the same base URL rules as the rest of the site (see api.js).
 */
import { ApiError, BASE, request } from './api.js';

export const TASKS = [
  {
    id: 'classification',
    endpoint: 'classification',
    mode: 'classifier',
    capability: 'classification',
    title: 'Seven-class classification',
    short: 'Classifier',
    blurb: 'The MobileNetV3 classifier alone on each labelled image (no YOLO, no binary gate) — the standard test-set evaluation.',
    format: 'classification',
    params: [],
  },
  {
    id: 'pipeline',
    endpoint: 'classification',
    mode: 'pipeline',
    capability: 'pipeline',
    title: 'End-to-end hybrid pipeline',
    short: 'Pipeline',
    blurb: 'YOLO → binary gate → classifier, scored per image against its label. Images with no retained object count as errors.',
    format: 'classification',
    params: ['conf', 'iou', 'max_det'],
  },
  {
    id: 'binary',
    endpoint: 'binary',
    capability: 'binary',
    title: 'Binary waste verification',
    short: 'Binary',
    blurb: 'The waste / non-waste MobileNetV3 gate at its fixed 0.35 threshold, against waste/ and non_waste/ labels.',
    format: 'binary',
    params: [],
  },
  {
    id: 'detection',
    endpoint: 'detection',
    capability: 'detection',
    title: 'YOLOv8 object detection',
    short: 'Detection',
    blurb: 'Precision, recall, mAP@0.50 and mAP@0.50:0.95 against YOLO-format bounding-box annotations.',
    format: 'detection',
    params: ['conf', 'iou', 'max_det'],
  },
  {
    id: 'inference',
    endpoint: 'inference',
    capability: 'inference',
    title: 'Inference speed benchmark',
    short: 'Speed',
    blurb: 'Times every pipeline stage on any images. No labels needed — and no accuracy is reported.',
    format: 'inference',
    params: ['conf', 'iou', 'max_det', 'gradcam'],
  },
];

async function json(path, { method = 'GET', body, timeoutMs = 30000 } = {}) {
  const init = { method };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { 'Content-Type': 'application/json' };
  }
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE}${path}`, { ...init, signal: ctrl.signal });
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      /* non-JSON */
    }
    if (!res.ok) {
      const d = data?.detail;
      throw new ApiError((typeof d === 'object' ? d?.message : d) || `Request failed (${res.status})`, {
        status: res.status,
        code: (typeof d === 'object' && d?.code) || 'http_error',
      });
    }
    return data;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Could not reach the EcoSortAI backend.', { code: 'offline' });
  } finally {
    clearTimeout(t);
  }
}

export const getSummary = () => request('/api/evaluation/summary', { timeoutMs: 30000 });
export const getJob = (id) => request(`/api/evaluation/jobs/${id}`, { timeoutMs: 15000 });
export const cancelJob = (id) => json(`/api/evaluation/jobs/${id}/cancel`, { method: 'POST' });
export const getRun = (id) => request(`/api/evaluation/runs/${id}`, { timeoutMs: 30000 });
export const getHistory = () => request('/api/evaluation/history', { timeoutMs: 15000 });
export const saveRun = (run_id, name) => json('/api/evaluation/history', { method: 'POST', body: { run_id, name } });
export const deleteRun = (id) => json(`/api/evaluation/history/${id}`, { method: 'DELETE' });
export const compareRuns = (ids) => request(`/api/evaluation/compare?ids=${ids.join(',')}`, { timeoutMs: 30000 });
export const exportUrl = (id, format, table = 'all') =>
  `${BASE}/api/evaluation/runs/${id}/export?format=${format}&table=${encodeURIComponent(table)}`;

/**
 * Upload a dataset and start a job. `files` is an array of {file, path}.
 * Uses XHR for upload progress. Resolves to {job_id}.
 */
export function startEvaluation(task, files, params, onUpload) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    const paths = [];
    for (const { file, path } of files) {
      fd.append('files', file, file.name);
      paths.push(path || file.name);
    }
    fd.append('paths', JSON.stringify(paths));
    if (task.mode) fd.append('mode', task.mode);
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') fd.append(k, String(v));
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${BASE}/api/evaluation/${task.endpoint}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onUpload?.(e.loaded / e.total);
    xhr.onerror = () => reject(new ApiError('Could not reach the EcoSortAI backend.', { code: 'offline' }));
    xhr.onload = () => {
      let data = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else {
        const d = data?.detail;
        reject(new ApiError((typeof d === 'object' ? d?.message : d) || `Request failed (${xhr.status})`, { status: xhr.status, code: d?.code }));
      }
    };
    xhr.send(fd);
  });
}
