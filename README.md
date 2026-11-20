# EcoSortAI — Cinematic Scroll-Based Waste Classification

An immersive, scroll-driven journey through the supplied eco-park footage, with seven
waste-category stations and a final **AI Waste Lab** that sends uploads to a real
PyTorch model (FastAPI) and shows its prediction, Grad-CAM, LIME and SHAP explanations
and disposal guidance.

```
ecosortai/
├─ frontend/                     React 18 + Vite 5 + GSAP 3 (ScrollTrigger)
│  ├─ public/media/
│  │  ├─ original/Camera_journey_through_eco-park_20261015162017.mp4   ← supplied video (untouched)
│  │  ├─ park-720.mp4 / park-480.mp4     all-intra H.264 web encodes of it (primary)
│  │  ├─ park-720.webm / park-480.webm   VP9 fallback for browsers without H.264
│  │  └─ poster.jpg, end.jpg, sample-metal.jpg (frames from the same video)
│  └─ src/
│     ├─ config/timeline.js      ★ station timestamps + bin boxes (edit here)
│     ├─ content/wasteCategories.js  educational panel content
│     ├─ lib/videoEngine.js      scroll → video time engine
│     ├─ components/             Journey, StationPanel, TopBar, WasteLab, ResultView…
│     └─ styles/global.css
└─ backend/                      FastAPI + PyTorch
   ├─ app/main.py                /api/health, /api/categories, /api/analyze
   ├─ app/pipeline.py            detect → classify → explain → recommend
   ├─ app/models/                checkpoint loader, optional YOLO, optional CLIP baseline
   ├─ app/xai/                   Grad-CAM, LIME, SHAP + overlays
   ├─ model/                     trained MobileNetV3 + YOLOv8 checkpoints and config
   └─ tests/test_api.py
```

---

## Quick start

Requirements: **Node 18+**, **Python 3.10–3.13**.

### Backend (Windows PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

macOS/Linux: same, with `source .venv/bin/activate`.
For an NVIDIA GPU install the CUDA build of PyTorch first (see pytorch.org), then the requirements.

### Frontend

```powershell
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api → http://127.0.0.1:8000)
```

Production: `npm run build` → static files in `frontend/dist/`. Serve them from any
static host and either reverse-proxy `/api` to the backend or build with
`VITE_API_BASE=https://your-api.example.com` (and add that site to `ECOSORT_CORS_ORIGINS`).

---

## Trained models included

This project ZIP now includes the supplied trained checkpoints and configuration:

- `backend/model/mobilenetv3_best.pth` — MobileNetV3 classifier, seven output classes.
- `backend/model/ecosortai_yolov8_best.pt` — YOLOv8 waste detector.
- `backend/model/model_config.json` — classifier architecture, class order and preprocessing settings.
- `backend/.env` — enables the custom classifier and YOLO detector for local development.

The backend uses YOLOv8 to propose object boxes and MobileNetV3 to classify each detected crop. If the detector finds no objects, the existing pipeline falls back to classifying the whole image. The model files are the uploaded trained weights; this integration does not retrain or alter them.

**Important model limitation:** the supplied detector previously produced some incorrect class predictions and broad boxes on sample images. Confidence thresholds affect which detections are returned but do not guarantee correctness. The integration is functional plumbing, not a claim that model accuracy has improved. Test it with representative images before relying on results.

The MobileNetV3 config assumes 224×224 input, ImageNet mean/std normalization, and direct resize to 224×224. Confirm these preprocessing choices against the original CNN training transform if its training script is available; a mismatch can affect predictions. The class order in the config is taken from the uploaded checkpoint metadata.

### Local startup

Install backend requirements (including Ultralytics for YOLO), then start the API:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Verify model status at `http://127.0.0.1:8000/api/health`; expect `model_loaded: true` and a detector entry. If `detector_error` is populated, check the backend terminal and installed `ultralytics` dependencies. For GPU acceleration, install a compatible CUDA build of PyTorch for your machine before starting the backend.

The current `.env` enables Grad-CAM, LIME and SHAP explanations. These can be computationally expensive, especially on CPU. For quicker initial checks, set `ECOSORT_EXPLAINERS=gradcam` in `backend/.env` and restart the API.

---

### Hybrid V2 inference integration

The backend now loads the supplied `binary_mobilenetv3_best.pth` checkpoint as a binary waste/non-waste gate. The analysis flow is:

1. YOLOv8 detection (`conf=0.05`, `iou=0.45`, `agnostic_nms=True`).
2. Crop each YOLO detection and compute the binary waste probability.
3. Reject a crop when `P(waste) < 0.35`.
4. Run the existing seven-class MobileNetV3 on retained crops.
5. Generate the configured XAI explanations and category disposal guidance.

The API health endpoint reports `binary_filter_loaded`, `binary_filter_error`, and `binary_threshold`. Each classified object includes `binary_waste_probability`; the result summary includes candidate and rejected counts. When YOLO is enabled and returns no candidates (or all candidates are rejected), the API returns zero classified objects rather than silently classifying the full scene.

The uploaded binary checkpoint is saved at `backend/model/binary_mobilenetv3_best.pth`. The included seven-class and YOLO checkpoints remain the versions already bundled in this website ZIP; they are not automatically replaced by other checkpoints stored in a separate Colab Drive folder. Confirm their versions against your latest Hybrid V2 notebook before final deployment.

---

## Performance Evaluation module

A dedicated **Performance Evaluation** page (`http://localhost:5173/#/evaluation`, linked as
**Performance** in the top bar) evaluates the complete system on *your* labelled test data:

| Section | What is measured | Needs |
|---|---|---|
| Classification | accuracy, per-class precision / recall / F1 / support, macro & weighted averages, 7×7 confusion matrix (counts, row %, column %) | images labelled by folder or `labels.csv` |
| End-to-end pipeline | the same metrics for YOLO → binary gate → classifier, per image; images where nothing is retained count as errors ("No prediction" column) | same |
| Binary verification | accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, 2×2 matrix, **waste acceptance rate** (TP / true waste) and **non-waste rejection rate** (TN / true non-waste) — reported separately from accuracy | `waste/` + `non_waste/` |
| YOLOv8 detection | precision, recall, mAP@0.50, mAP@0.50:0.95, per class + class-agnostic | YOLO `images/` + `labels/` + `data.yaml` |
| Inference speed | mean / median / min / max / p95 per image and per stage (preprocessing, YOLO, binary filter, CNN, Grad-CAM, recommendation), FPS | any images |

Honesty rules built into the code:

- No metric is hard-coded. Every number comes from model predictions compared with ground truth in that run.
- Unlabelled uploads never produce accuracy / precision / recall / F1 / mAP — the page shows a warning and only timing.
- mAP is only computed from bounding-box annotations, never from classification labels or YOLO confidence.
- Detection matching and AP use the installed Ultralytics functions (`BaseValidator.match_predictions`, `utils.metrics.ap_per_class`), with a NumPy port as fallback; both are cross-checked in the tests.
- Each run lists exactly which stages its timing includes (LIME/SHAP and HTTP transfer are never included). FPS = 1000 / mean end-to-end latency, batch size 1, after warm-up.
- The metrics stored inside the checkpoints (classifier `best_val_accuracy` 0.9151 at epoch 66; binary `best_val_balanced_accuracy` 0.9644 at epoch 2; YOLO `train_metrics` P 0.753 / R 0.743 / mAP50 0.772 / mAP50-95 0.630 written by Ultralytics 8.4.177) are shown in a separate **"Historical metrics recorded in the checkpoints"** table with their source and the note that the dataset and sample counts were not recorded. They are never mixed with measured runs.
- The inference pipeline is unchanged: binary threshold 0.35, YOLO conf 0.05, IoU 0.45, agnostic NMS. Evaluation reuses the already-loaded model instances; a shared lock stops evaluation and live analysis from running a model at the same time.

Runs can be **saved** (results only → `backend/evaluation_runs/<run_id>.json`), compared (two or more), and exported as a **PDF report** or **CSV** tables. Uploaded datasets are extracted to a temporary folder and deleted when the run finishes.

### Windows setup (PowerShell) — run backend and frontend in two terminals

Terminal 1 — backend:

```powershell
cd ecosortai\backend
py -3.11 -m venv .venv            # Python 3.10–3.13 all work
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt   # includes reportlab + pyyaml for the evaluation module
uvicorn app.main:app --port 8000
```

Check `http://127.0.0.1:8000/api/health` (expect `model_loaded: true`, `binary_filter_loaded: true`, a `detector` entry)
and `http://127.0.0.1:8000/api/evaluation/summary`. Interactive API docs: `http://127.0.0.1:8000/docs`.

Terminal 2 — frontend:

```powershell
cd ecosortai\frontend
npm install                       # adds plotly.js-dist-min (loaded only on the evaluation page)
npm run dev                       # http://localhost:5173  →  Performance  (or /#/evaluation)
```

Run the tests (backend venv active): `python -m pytest -q tests`.

### Using the page

1. Choose a task. Tasks whose model is not loaded are disabled.
2. Choose a dataset: a ZIP, a folder (Chrome / Edge folder picker) or loose images. Formats are shown under the drop zone; examples are in `backend/sample_eval_data/` (see its README).
3. Set parameters (YOLO confidence / NMS IoU / max boxes for pipeline, detection and speed tasks; warm-up images). Detection defaults to confidence 0.001 like `yolo val`, so the whole precision–recall curve is measured; the live site keeps 0.05.
4. Start. Progress is polled; you can cancel. Results appear below; **Save run**, **PDF report**, **CSV**.
5. Evaluation history: tick two or more runs → **Compare selected**.

### Evaluation API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/evaluation/summary` | models, checkpoint sha256, thresholds, capabilities, historical checkpoint metrics |
| POST | `/api/evaluation/classification` | multipart `files[]` (+ `paths` JSON), `mode=classifier\|pipeline`, `conf`, `iou`, `max_det`, `warmup`, `dataset_name` → `{job_id}` |
| POST | `/api/evaluation/binary` | `files[]`, `warmup` |
| POST | `/api/evaluation/detection` | `files[]`, `conf`, `iou`, `max_det`, `warmup` |
| POST | `/api/evaluation/inference` | `files[]`, `conf`, `iou`, `max_det`, `warmup`, `gradcam` |
| GET | `/api/evaluation/jobs/{job_id}` · POST `…/cancel` | progress |
| GET | `/api/evaluation/runs/{run_id}` | full result |
| GET / POST | `/api/evaluation/history` | list / save `{run_id, name}` |
| DELETE | `/api/evaluation/history/{run_id}` | delete a saved run |
| GET | `/api/evaluation/compare?ids=a,b` | side-by-side summary |
| GET | `/api/evaluation/runs/{run_id}/export?format=pdf\|csv&table=…` · POST `/api/evaluation/export` | downloads |

Optional `.env` settings: `ECOSORT_EVAL_MAX_DATASET_MB` (2048), `ECOSORT_EVAL_MAX_FILES` (20000), `ECOSORT_EVAL_RUNS_DIR`.
Existing endpoints (`/api/health`, `/api/categories`, `/api/analyze`) are unchanged.

### Evaluation limitations

- The bundled sample datasets only demonstrate the formats; their scores are not meaningful (see `backend/sample_eval_data/README.md`).
- Binary evaluation feeds each whole image to the gate; in the live pipeline it sees YOLO crops. Evaluate with crop-like images for the most representative numbers.
- Detection predictions use `YOLO.predict()` per image (letterboxed, batch 1); `yolo val` uses rectangular batches, so mAP can differ slightly.
- Timings are measured in the API process on whatever device it uses (CPU here); they exclude network upload.
- Unsaved runs live in memory and disappear when the API restarts.

## How the scroll journey works

- One sticky full-viewport stage holds the video. A single GSAP **ScrollTrigger** maps
  page progress → "scroll units"; a `gsap.ticker` loop smooths the target time and
  seeks the video (never queuing a seek while one is in flight). React state changes
  only when the station/phase changes — no per-scroll re-renders.
- The whole video is fetched into memory (Blob URL, with a real progress bar) before
  scrubbing, and the encodes are **all-intra** so every frame seeks instantly.
- Each station = **approach** (camera moves to the bin) → **hold** on the inspection
  frame (spotlight, reticle, object chips, gentle push-in, panel in → read → out) →
  **depart** (camera continues). Works identically scrolling up or down.
- Panels sit on the side opposite the bin, so they never cover it; on phones the frame pans
  horizontally to keep the bin centred and the panel becomes a compact top sheet.
- **Mobile fallback:** if seeks are measured to be slow (median > 160 ms), the engine
  switches to rate-controlled playback toward the target time. Force it with `?mode=playback`.
- **Reduced motion:** no camera travel or zoom — the view cuts between inspection
  frames, and transitions are disabled.
- Navigation: station indicator (`03 / 07`), progress bar, station dots, prev/next,
  *Skip to Waste Lab*, restart, keyboard-accessible tabs.

### Video timeline (from frame inspection)

| # | Category | Bin in footage | side | approach → **inspect** → depart (s) |
|---|---|---|---|---|
| 1 | Plastic | blue wheelie bin, lid propped open, PET bottles | L | 0.00 → **1.80** → 2.30 |
| 2 | Paper | green bin with newspapers & cardboard | R | 2.30 → **2.80** → 3.30 |
| 3 | Metal | galvanised can full of aluminium cans | L | 3.30 → **4.60** → 4.90 |
| 4 | Organic | brown bin heaped with food scraps | R | 4.90 → **5.05** → 5.55 |
| 5 | E-Waste | charcoal bin, lid open, electronic scrap | L | 5.55 → **6.00** → 6.50 |
| 6 | Hazardous | red bin, skull-and-crossbones labels, lid open | L | 6.50 → **6.95** → 7.25 |
| 7 | Glass | transparent bin with green/clear bottles | R | 7.25 → **7.45** → 7.85 |
| — | End of path → Waste Lab | path continues to the gazebo | — | 7.85 → 7.96 |

The footage contains **no lid-opening action** (the blue, e-waste and hazardous lids are
already open in every frame). Rather than faking one, the site uses a light spotlight,
detection-style reticle and object chips aligned to each bin. All values are editable in
`frontend/src/config/timeline.js`.

### What is educational vs. real

- Station panels are **predefined educational content**. Anything resembling model
  output there (example class/confidence, Grad-CAM/LIME/SHAP diagrams) is labelled
  *illustrative* and drawn as schematics.
- When a model is loaded, each panel's *Explainable AI* tab offers **"Analyse this video
  frame"**, which sends the bin crop to the real model and shows its genuine output.
- In the Lab, **"View a sample result (demo data)"** shows the layout with a
  `DEMO DATA` banner; everything else in the Lab comes from the backend.

---

## Testing performed

- `pytest backend/tests` — 18 tests (13 new for the evaluation module, see `EVALUATION_CHANGES.md`). The original 5: honest no-model 503, input validation, full response contract with Grad-CAM/LIME/SHAP images, multi-object detection path, uncertain fallback. These use an **untrained** resnet18 (`tools/make_pipeline_test_model.py`) and check plumbing only, not accuracy.
- Headless Chromium (Playwright): desktop 1440×900 and mobile 390×844 — intro, all 7 stations (approach + inspect), outro, Lab hand-off, reverse scrolling (lands on the exact frame), reduced motion, forced playback mode, no console errors from the app.
- End-to-end upload through the real API (untrained test model): validation error for a `.txt`, scanning state, result with model-generated Grad-CAM/LIME/SHAP, correctly reported as *Uncertain* (22% top score); multi-object bounding boxes; error + retry state; station-panel live frame analysis.

Not yet tested: real predictions (needs your trained model), YOLO detection with real
weights, the CLIP baseline, and physical iOS/Android devices.

