# Performance Evaluation upgrade — changes, tests, limitations

## New files

Backend
- `backend/app/evaluation/__init__.py`
- `backend/app/evaluation/api.py` — `/api/evaluation/*` router (jobs, runs, history, compare, export)
- `backend/app/evaluation/dataset.py` — safe ZIP/folder ingestion (zip-slip, size and file-count limits, temp dir), label discovery (folder-per-class, `labels.csv`, YOLO txt + `data.yaml`), image validation
- `backend/app/evaluation/metrics.py` — classification report, confusion matrix, binary metrics + acceptance/rejection rates + ROC-AUC, YOLO matching/AP (Ultralytics, with NumPy port fallback), timing stats
- `backend/app/evaluation/runner.py` — stage-timed replica of the hybrid pipeline (same components, thresholds, order) and the four task runners
- `backend/app/evaluation/store.py` — background job queue, run cache, saved history, checkpoint sha256 identifiers, historical checkpoint metrics
- `backend/app/evaluation/report.py` — PDF report (reportlab) and CSV tables, observations derived from the run's numbers
- `backend/app/runtime.py` — shared model lock
- `backend/tests/test_evaluation.py` — 13 tests
- `backend/sample_eval_data/` — example datasets for every format + README

Frontend
- `frontend/src/components/evaluation/EvaluationPage.jsx` — page, run form, KPI row, history & comparison
- `frontend/src/components/evaluation/sections.jsx` — classification, binary, detection, inference, historical sections
- `frontend/src/components/evaluation/Plot.jsx` — lazy Plotly wrapper + chart tokens
- `frontend/src/lib/evalApi.js` — API client (XHR upload with progress)
- `frontend/src/styles/evaluation.css`

Docs
- `EVALUATION_CHANGES.md` (this file)

## Modified files

| File | Change |
|---|---|
| `backend/app/main.py` | includes the evaluation router; `/api/analyze` now runs inside the shared model lock (same function, same arguments); CORS allows `DELETE`; unreadable images that make newer Pillow try to load an optional plugin (`ImportError`) now return the normal 400 `invalid_image` instead of a 500 |
| `backend/.env` | **bug fix:** the file contained literal `\n` sequences, so `ECOSORT_DETECTOR_CONF` was read as `0.05\nECOSORT_DETECTOR_IOU=0.45`, which crashed start-up, and the binary-checkpoint/threshold lines were ignored. Split onto separate lines; values unchanged (conf 0.05, IoU 0.45, binary 0.35) |
| `backend/.env.example` | optional evaluation settings |
| `backend/requirements.txt` | `reportlab`, `pyyaml`, `scikit-learn` (tests only) |
| `backend/tests/test_api.py` | the plumbing tests now disable the bundled YOLO/binary models that `.env` enables, so they test what they were written for (they failed once `.env` loaded correctly) |
| `frontend/src/App.jsx` | `#/evaluation` route; the journey component is unchanged (renamed `JourneyApp` inside the file); evaluation page is lazy-loaded |
| `frontend/src/components/TopBar.jsx` | **Performance** link |
| `frontend/src/lib/hooks.js` | `useHashRoute()` |
| `frontend/src/lib/api.js` | exports `BASE` and `request` for reuse |
| `frontend/src/styles/global.css` | styles for the top-bar link and route loading text |
| `frontend/package.json`, `package-lock.json` | `plotly.js-dist-min` 3.7.0 |
| `README.md` | Performance Evaluation section, Windows setup, API table, limitations |
| `.gitignore` | `backend/evaluation_runs/` |

Not modified: the models and checkpoints, `pipeline.py`, `models/*`, `xai/*`, `categories.py`, the scroll journey, Waste Lab and station components.

## Tests run

**pytest** (`python -m pytest -q tests`): **18 passed**.
- Metric code vs scikit-learn `classification_report` / `confusion_matrix` / `roc_auc_score` (identical).
- Detection: matching, IoU and AP port vs installed Ultralytics 8.4.212 (identical); perfect detector → mAP 0.995 (Ultralytics' 101-point value); no GT boxes → no mAP.
- `timed_pipeline` gives exactly the boxes, categories, confidences, uncertainty and rejections of `pipeline.analyze`.
- API: labelled ZIP → metrics, unmapped label folder reported, corrupt image listed as invalid, zip-slip entry ignored; unlabelled images → no supervised metrics + warning; folder upload with relative paths; binary and detection jobs with stub models (known TP/FP/TN/FN and AP); missing models → 409, bad ZIP → 400, bad mode → 422; save / history / compare / delete; PDF and CSV export; `/api/analyze` still works.
- Original 5 API tests still pass.

**Real checkpoints, sample datasets** (CPU, this build machine). These verify the workflow; the images are AI-rendered bins, not the models' domain, so the scores are not meaningful:

| Run | Images | Result |
|---|---|---|
| Seven-class classifier | 21 | accuracy 19.1%, macro F1 0.133; mean 13.5 ms / image |
| End-to-end pipeline | 21 | accuracy 23.8%, macro F1 0.163, 1 image with no retained object; mean 107 ms |
| Binary verification | 22 | accuracy 68.2%, waste acceptance 100% (14/14), non-waste rejection 12.5% (1/8) |
| YOLO detection | 7 (bin boxes) | mAP@0.50 0.000 class-aware (class-agnostic 0.010) — the GT boxes are whole bins, not the objects YOLO was trained on |
| Inference benchmark (Grad-CAM on) | 3 | mean 287 ms: preprocessing 6, YOLO 76, binary 24, CNN 18, Grad-CAM 162, recommendation 0.05 ms → 3.5 FPS |
| Unlabelled → classification | 3 | warning shown, no accuracy/precision/recall/F1 |

**Browser** (headless Chrome, 1440×900 and 390×844, production build): journey page loads with the new Performance link; evaluation page; all five tasks run from the UI with ZIP uploads; confusion-matrix normalisation toggle; save run; compare two runs; back to the journey; mobile layout. No console errors.
`/api/analyze` on `sample-metal.jpg` with the real models: 200, Grad-CAM generated, disposal recommendation returned (existing behaviour; the bundled model labels that image Organic, which was already noted as a model limitation).
`npm run build` and `npm run build:single` both succeed.

## Remaining limitations

- No real labelled test split was available here, so no meaningful accuracy for your models has been measured yet — run the page on your held-out test set.
- Not tested on a physical Windows machine; commands follow the existing README. Folder picking needs Chrome/Edge (Firefox: use a ZIP).
- `npm run build:single` now also inlines Plotly (≈ +5 MB). The normal build loads Plotly only on the evaluation page.
- Binary evaluation uses whole images, detection uses `predict()` rather than `yolo val` batching, and unsaved runs are lost on API restart (details in README).
- Python 3.13 / Pillow 13 / Ultralytics 8.4.212 / NumPy 2 were used for testing.
