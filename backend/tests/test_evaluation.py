"""Tests for the Performance Evaluation module.

* Metric functions are cross-checked against scikit-learn and Ultralytics.
* The instrumented evaluation runner must give exactly the predictions of
  ``pipeline.analyze`` (it only adds per-stage timing).
* API workflow: labelled → metrics; unlabelled → no metrics + warning;
  invalid files reported; history; PDF/CSV export.
Models here are UNTRAINED test models — the numbers check plumbing, not quality.
"""
import io
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from test_api import _client, _png  # noqa: E402  (reuses the API test fixture helpers)


# --------------------------------------------------------------------------- metrics
def _metrics():
    from app.evaluation import metrics

    return metrics


def test_classification_report_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    m = _metrics()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 300)
    p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 7, 300))
    rep = m.classification_report(y.tolist(), p.tolist(), list("abcdefg"), list("ABCDEFG"))
    ref = skm.classification_report(y, p, labels=list(range(7)), output_dict=True, zero_division=0)
    assert rep["accuracy"] == pytest.approx(ref["accuracy"], abs=1e-4)
    for i, row in enumerate(rep["per_class"]):
        r = ref[str(i)]
        assert row["precision"] == pytest.approx(r["precision"], abs=1e-4)
        assert row["recall"] == pytest.approx(r["recall"], abs=1e-4)
        assert row["f1"] == pytest.approx(r["f1-score"], abs=1e-4)
        assert row["support"] == r["support"]
    for k, kk in (("macro_avg", "macro avg"), ("weighted_avg", "weighted avg")):
        assert rep[k]["f1"] == pytest.approx(ref[kk]["f1-score"], abs=1e-4)
        assert rep[k]["precision"] == pytest.approx(ref[kk]["precision"], abs=1e-4)
    assert np.array(rep["confusion_matrix"]).tolist() == skm.confusion_matrix(y, p, labels=list(range(7))).tolist()


def test_abstentions_count_as_errors_and_absent_classes_are_excluded():
    m = _metrics()
    rep = m.classification_report([0, 0, 1, 1], [0, None, 1, 0], ["a", "b", "c"], ["A", "B", "C"])
    assert rep["accuracy"] == 0.5 and rep["n_abstained"] == 1 and rep["coverage"] == 0.75
    assert rep["per_class"][0]["recall"] == 0.5 and rep["per_class"][0]["abstained"] == 1
    assert rep["per_class"][2]["support"] == 0 and rep["per_class"][2]["recall"] is None
    assert rep["averaged_over_classes"] == ["a", "b"]


def test_binary_report_rates_are_class_conditional():
    m = _metrics()
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    pr = [0.9, 0.8, 0.36, 0.2, 0.1, 0.5, 0.0, 0.34, 0.2, 0.1]
    b = m.binary_report(y, pr, 0.35)
    assert (b["tp"], b["fn"], b["fp"], b["tn"]) == (3, 1, 1, 5)
    assert b["accuracy"] == 0.8
    assert b["waste_acceptance_rate"] == 0.75 and b["recall"] == 0.75
    assert b["non_waste_rejection_rate"] == pytest.approx(5 / 6, abs=1e-4)
    assert b["overall_acceptance_rate"] == 0.4  # differs from accuracy on purpose
    skm = pytest.importorskip("sklearn.metrics")
    assert b["roc_auc"] == pytest.approx(skm.roc_auc_score(y, pr), abs=1e-4)


def test_ap_matches_ultralytics():
    um = pytest.importorskip("ultralytics.utils.metrics")
    m = _metrics()
    rng = np.random.default_rng(1)
    n = 400
    tp = rng.random((n, 10)) < np.linspace(0.8, 0.2, 10)
    conf = rng.random(n)
    pred_cls = rng.integers(0, 4, n)
    target_cls = rng.integers(0, 4, 300)
    mine = m.ap_per_class(tp.astype(float), conf, pred_cls, target_cls)
    ref = um.ap_per_class(tp, conf, pred_cls, target_cls)
    # ultralytics returns tp, fp, p, r, f1, ap, unique_classes, ...
    p_ref, r_ref, ap_ref = ref[2], ref[3], ref[5]
    np.testing.assert_allclose(mine["ap"], ap_ref, atol=1e-6)
    np.testing.assert_allclose(mine["p"], p_ref, atol=1e-6)
    np.testing.assert_allclose(mine["r"], r_ref, atol=1e-6)


def test_detection_accumulator_perfect_and_empty():
    m = _metrics()
    acc = m.DetectionAccumulator(["a", "b"])
    acc.add([[0, 0, 10, 10], [20, 20, 40, 40]], [0.9, 0.8], [0, 1], [[0, 0, 10, 10], [20, 20, 40, 40]], [0, 1])
    rep = acc.report()["class_aware"]
    # Ultralytics' 101-point interpolation gives 0.995 for a perfect detector
    assert rep["map50"] == pytest.approx(0.995, abs=1e-3) and rep["map50_95"] == pytest.approx(0.995, abs=1e-3)
    acc2 = m.DetectionAccumulator(["a"])
    acc2.add([[0, 0, 10, 10]], [0.9], [0], [], [])
    assert acc2.report()["class_aware"]["available"] is False  # no GT → no mAP


def test_matching_port_matches_ultralytics():
    pytest.importorskip("ultralytics")
    m = _metrics()
    rng = np.random.default_rng(3)
    for _ in range(20):
        g = rng.integers(1, 6)
        n = rng.integers(1, 9)
        gt = rng.random((g, 2)) * 50
        gt = np.concatenate([gt, gt + 10 + rng.random((g, 2)) * 20], 1)
        pr = gt[rng.integers(0, g, n)] + rng.normal(0, 4, (n, 4))
        iou = m.box_iou(gt, pr)
        gk, pk = rng.integers(0, 2, g), rng.integers(0, 2, n)
        ref, impl = m._match(pk, gk, iou)
        assert impl == "ultralytics"
        np.testing.assert_array_equal(m.match_predictions(pk, gk, iou), ref)


def test_iou_matches_ultralytics():
    um = pytest.importorskip("ultralytics.utils.metrics")
    import torch

    m = _metrics()
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 25]], float)
    b = np.array([[0, 0, 10, 10], [6, 4, 14, 20], [30, 30, 40, 40]], float)
    np.testing.assert_allclose(m.box_iou(a, b), um.box_iou(torch.tensor(a), torch.tensor(b)).numpy(), atol=1e-6)


# --------------------------------------------------------------------------- dataset helpers
def _img_bytes(color, size=(96, 80)):
    arr = np.zeros((size[1], size[0], 3), np.uint8)
    arr[:] = color
    arr[20:60, 20:70] = (230, 230, 230)
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG")
    return b.getvalue()


def _zip(entries: dict[str, bytes]) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in entries.items():
            z.writestr(k, v)
    return b.getvalue()


def _wait(c, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/evaluation/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job timed out")


def _start(c, ep, files, **data):
    r = c.post(f"/api/evaluation/{ep}", files=files, data=data)
    assert r.status_code == 200, r.text
    j = _wait(c, r.json()["job_id"])
    return j, (c.get(f"/api/evaluation/runs/{j['run_id']}").json() if j["run_id"] else None)


# --------------------------------------------------------------------------- runner equivalence
def test_timed_pipeline_matches_pipeline_analyze(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:  # noqa: F841
        from app.evaluation.runner import Models, timed_pipeline
        from app.models.detector import Detection
        from app.models.registry import registry
        from app.pipeline import analyze

        class StubDetector:
            name, conf, iou, max_det = "stub", 0.05, 0.45, 30

            def detect(self, _img):
                return [Detection((10, 10, 150, 200), "plastic", 0.9), Detection((170, 40, 310, 230), "metal", 0.8)]

        class StubBinary:
            threshold = 0.35

            def predict_waste_probability(self, crops):
                return np.array([0.9, 0.2][: len(crops)])

        p = tmp_path / "x.png"
        p.write_bytes(_png())
        img = Image.open(p).convert("RGB")
        ref = analyze(img, registry.classifier, StubDetector(), ("gradcam",), StubBinary())
        got = timed_pipeline(p, Models(registry.classifier, StubDetector(), StubBinary()), conf=0.05, iou=0.45, max_det=30, gradcam=True)
        assert len(got["objects"]) == len(ref["objects"]) == 1 and got["n_rejected"] == ref["summary"]["rejected_count"] == 1
        o, r = got["objects"][0], ref["objects"][0]
        assert o["bbox"] == r["bbox"] and o["confidence"] == r["prediction"]["confidence"]
        assert o["top_category"] == r["prediction"]["top_k"][0]["category"]
        assert o["uncertain"] == r["prediction"]["uncertain"]
        assert set(got["timing_ms"]) == {"preprocessing", "detection", "binary_filter", "classification", "gradcam", "recommendation"}
        assert all(v is not None and v >= 0 for v in got["timing_ms"].values())


# --------------------------------------------------------------------------- API workflow
def test_classification_workflow_with_labels(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:
        s = c.get("/api/evaluation/summary").json()
        assert s["capabilities"]["classification"] is True and s["capabilities"]["detection"] is False
        entries = {f"ds/{cls}/{i}.jpg": _img_bytes((30 * k, 80, 200 - 20 * k)) for k, cls in enumerate(["plastic", "glass", "e-waste"]) for i in range(3)}
        entries["ds/metal/broken.jpg"] = b"not an image"
        entries["ds/textile/0.jpg"] = _img_bytes((1, 2, 3))  # unmapped label
        entries["../evil.jpg"] = _img_bytes((0, 0, 0))  # zip-slip must be ignored
        j, run = _start(c, "classification", [("files", ("ds.zip", _zip(entries), "application/zip"))], mode="classifier")
        assert j["status"] == "done", j
        cl = run["classification"]
        assert run["supervised"] is True and cl["n_samples"] == 9
        assert sum(sum(r) for r in cl["confusion_matrix"]) == 9 and len(cl["confusion_matrix"]) == 7
        assert {r["label"]: r["support"] for r in cl["per_class"]}["E-Waste"] == 3
        assert run["dataset"]["n_invalid"] == 1 and run["dataset"]["unmapped_labels"] == {"textile": 1}
        assert any("could not be mapped" in w for w in run["warnings"])
        inc = run["inference"]["includes"]
        assert inc["classification"] and not inc["detection"] and not inc["gradcam"]
        assert run["inference"]["fps"] > 0 and run["inference"]["end_to_end"]["n"] == 10

        # save, history, compare, export
        assert c.post("/api/evaluation/history", json={"run_id": run["run_id"], "name": "t1"}).json()["saved"]
        _, run2 = _start(c, "classification", [("files", ("ds.zip", _zip(entries), "application/zip"))], mode="classifier")
        h = c.get("/api/evaluation/history").json()["runs"]
        assert {x["run_id"] for x in h} >= {run["run_id"], run2["run_id"]}
        cmp_ = c.get(f"/api/evaluation/compare?ids={run['run_id']},{run2['run_id']}").json()
        assert len(cmp_["runs"]) == 2 and cmp_["runs"][0]["per_class_f1"]
        pdf = c.get(f"/api/evaluation/runs/{run['run_id']}/export?format=pdf")
        assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
        csv = c.post("/api/evaluation/export", json={"run_id": run["run_id"], "format": "csv", "table": "confusion_matrix"})
        assert csv.status_code == 200 and "true \\ predicted" in csv.text and "Paper & Cardboard" in csv.text
        assert c.delete(f"/api/evaluation/history/{run['run_id']}").status_code == 200


def test_unlabeled_upload_reports_no_supervised_metrics(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:
        files = [("files", (f"img{i}.jpg", _img_bytes((i * 40, 90, 90)), "image/jpeg")) for i in range(3)]
        j, run = _start(c, "classification", files, mode="classifier")
        assert j["status"] == "done"
        assert run["supervised"] is False and "classification" not in run
        assert any("NOT reported" in w for w in run["warnings"])
        assert run["inference"]["end_to_end"]["n"] == 3


def test_folder_upload_with_paths(monkeypatch, tmp_path):
    import json

    with _client(monkeypatch, tmp_path, with_model=True) as c:
        names = ["set/waste/a.jpg", "set/waste/b.jpg", "set/non_waste/c.jpg"]
        files = [("files", (n.split("/")[-1], _img_bytes((k * 60, 50, 50)), "image/jpeg")) for k, n in enumerate(names)]
        r = c.post("/api/evaluation/classification", files=files, data={"paths": json.dumps(names), "mode": "classifier"})
        j = _wait(c, r.json()["job_id"])
        run = c.get(f"/api/evaluation/runs/{j['run_id']}").json()
        assert run["dataset"]["name"] == "set" and run["dataset"]["source"] == "folder"
        # binary filter is disabled in the test client → 409, not fake numbers
        r = c.post("/api/evaluation/binary", files=files, data={"paths": json.dumps(names)})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "binary_not_loaded"
        r = c.post("/api/evaluation/detection", files=files)
        assert r.status_code == 409


def test_binary_and_detection_with_stub_models(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:
        from app.models.detector import Detection
        from app.models.registry import registry

        class StubBinary:
            threshold = 0.35
            class_names = ["non_waste", "waste"]

            def predict_waste_probability(self, crops):
                # bright red channel → "waste"
                return np.array([float(np.asarray(im)[..., 0].mean() > 100) * 0.9 + 0.05 for im in crops])

        class _Names:
            names = {0: "plastic", 1: "metal"}

        class StubDetector:
            name, conf, iou, max_det = "stub", 0.001, 0.45, 300
            model = _Names()

            def detect(self, img):
                c = 0.9 if np.asarray(img).mean() > 50 else 0.5
                return [Detection((20, 20, 70, 60), "plastic", c), Detection((0, 0, 10, 10), "metal", 0.3)]

        registry.binary_filter = StubBinary()
        registry.detector = StubDetector()
        try:
            ent = {"b/waste/1.jpg": _img_bytes((200, 10, 10)), "b/waste/2.jpg": _img_bytes((210, 10, 10)),
                   "b/non_waste/1.jpg": _img_bytes((10, 10, 200)), "b/non_waste/2.jpg": _img_bytes((150, 10, 10))}
            _, run = _start(c, "binary", [("files", ("b.zip", _zip(ent), "application/zip"))])
            b = run["binary"]
            assert (b["tp"], b["fn"], b["tn"], b["fp"]) == (2, 0, 1, 1)
            assert b["waste_acceptance_rate"] == 1.0 and b["non_waste_rejection_rate"] == 0.5 and b["accuracy"] == 0.75

            # YOLO-format detection: box (20,20,70,60) on 96×80 image, class "plastic"
            lab = f"0 {45 / 96:.6f} {40 / 80:.6f} {50 / 96:.6f} {40 / 80:.6f}\n"
            ent = {"d/images/a.jpg": _img_bytes((90, 90, 90)), "d/labels/a.txt": lab.encode(),
                   "d/images/b.jpg": _img_bytes((5, 5, 5)), "d/labels/b.txt": b"0 2 2 2\nbad line\n",
                   "d/data.yaml": b"names: [plastic]\n"}
            _, run = _start(c, "detection", [("files", ("d.zip", _zip(ent), "application/zip"))], conf="0.001", iou="0.45", max_det="300")
            d = run["detection"]
            assert d["n_ground_truth_boxes"] == 1 and d["n_predicted_boxes"] == 4
            # the single GT is matched by the highest-confidence prediction → AP = 0.995 (101-point interpolation)
            assert d["class_aware"]["map50"] == pytest.approx(0.995, abs=1e-3)
            assert run["dataset"]["label_problems"], "invalid label lines must be reported"

            # no labels/ folder → no mAP
            _, run = _start(c, "detection", [("files", ("u.zip", _zip({"u/images/a.jpg": _img_bytes((5, 5, 5))}), "application/zip"))],
                            conf="0.001", iou="0.45", max_det="300")
            assert run["supervised"] is False and "detection" not in run
        finally:
            registry.binary_filter = None
            registry.detector = None


def test_bad_uploads_are_rejected_cleanly(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, with_model=True) as c:
        r = c.post("/api/evaluation/classification", files=[("files", ("x.zip", b"not a zip", "application/zip"))])
        assert r.status_code == 400 and r.json()["detail"]["code"] == "invalid_zip"
        j, _ = (lambda r: (_wait(c, r.json()["job_id"]), None))(
            c.post("/api/evaluation/classification", files=[("files", ("e.zip", _zip({"readme.md": b"hi"}), "application/zip"))]))
        assert j["status"] == "error" and j["error"]["code"] == "no_images"
        r = c.post("/api/evaluation/classification", files=[("files", ("a.jpg", _img_bytes((1, 1, 1)), "image/jpeg"))], data={"mode": "weird"})
        assert r.status_code == 422
        assert c.get("/api/evaluation/runs/zzz").status_code == 404
        # existing endpoint still works
        assert c.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")}, data={"explain": "gradcam"}).status_code == 200
