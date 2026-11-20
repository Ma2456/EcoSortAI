"""PDF and CSV exports of an evaluation run. Every number comes from the run."""
from __future__ import annotations

import csv
import io

from .runner import STAGE_LABELS, STAGES


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.2f}%"


def _num(v, nd=3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _ms(v) -> str:
    return "—" if v is None else f"{v:.1f} ms"


# --------------------------------------------------------------------------- observations
def observations(run: dict) -> list[str]:
    """Factual statements derived from the run's numbers (no interpretation beyond them)."""
    obs: list[str] = []
    c = run.get("classification")
    if c:
        rows = [r for r in c["per_class"] if r["support"] > 0 and r["f1"] is not None]
        if rows:
            best = max(rows, key=lambda r: r["f1"])
            worst = min(rows, key=lambda r: r["f1"])
            obs.append(f"Highest per-class F1: {best['label']} ({best['f1']:.3f}, n={best['support']}); "
                       f"lowest: {worst['label']} ({worst['f1']:.3f}, n={worst['support']}).")
        cm = c["confusion_matrix"]
        off = [(cm[i][j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i][j] > 0]
        if off:
            v, i, j = max(off)
            obs.append(f"Most frequent confusion: true {c['label_names'][i]} predicted as {c['label_names'][j]} ({v} image(s)).")
        if c.get("n_abstained"):
            obs.append(f"{c['n_abstained']} image(s) received no prediction (no object retained by the pipeline) and are counted as errors.")
        small = [r["label"] for r in c["per_class"] if 0 < r["support"] < 30]
        if small:
            obs.append(f"Classes with fewer than 30 test images ({', '.join(small)}) have wide uncertainty on their per-class metrics.")
        missing = [r["label"] for r in c["per_class"] if r["support"] == 0]
        if missing:
            obs.append(f"No test images for: {', '.join(missing)} — these classes are not evaluated and are excluded from the averages.")
    b = run.get("binary")
    if b:
        obs.append(f"Binary gate at threshold {b['threshold']}: waste acceptance rate {_pct(b['waste_acceptance_rate'])}, "
                   f"non-waste rejection rate {_pct(b['non_waste_rejection_rate'])} (accuracy {_pct(b['accuracy'])}).")
    d = (run.get("detection") or {}).get("class_aware")
    if d and d.get("available"):
        agn = run["detection"]["class_agnostic"]
        obs.append(f"Detection mAP@0.50 is {_num(d['map50'])} class-aware vs {_num(agn.get('map50'))} class-agnostic; "
                   "the gap reflects class confusion by the detector rather than localisation.")
    inf = run.get("inference") or {}
    st = inf.get("stages") or {}
    if st:
        k = max(st, key=lambda s: st[s]["mean"])
        e2e = inf.get("end_to_end") or {}
        share = st[k]["mean"] / e2e["mean"] if e2e.get("mean") else None
        obs.append(f"Slowest stage: {STAGE_LABELS[k]} (mean {_ms(st[k]['mean'])}"
                   + (f", {share:.0%} of end-to-end time" if share else "") + ").")
    return obs


def limitations(run: dict) -> list[str]:
    out = list(run.get("notes", []))
    out += [w for w in run.get("warnings", []) if w not in out]
    out.append("Metrics describe this dataset only; they are not a guarantee of performance on other images.")
    if (run.get("inference") or {}).get("device", "").startswith("cpu"):
        out.append("Timings were measured on CPU in this server process; GPU timings will differ.")
    return out


# --------------------------------------------------------------------------- CSV
def csv_tables(run: dict) -> dict[str, list[list]]:
    t: dict[str, list[list]] = {}
    c = run.get("classification")
    if c:
        rows = [["category", "precision", "recall", "f1", "support", "predicted", "abstained"]]
        for r in c["per_class"]:
            rows.append([r["label"], r["precision"], r["recall"], r["f1"], r["support"], r["predicted"], r["abstained"]])
        rows.append(["macro avg", c["macro_avg"]["precision"], c["macro_avg"]["recall"], c["macro_avg"]["f1"], c["n_samples"], "", ""])
        rows.append(["weighted avg", c["weighted_avg"]["precision"], c["weighted_avg"]["recall"], c["weighted_avg"]["f1"], c["n_samples"], "", ""])
        rows.append(["accuracy", "", "", c["accuracy"], c["n_samples"], "", ""])
        t["classification_report"] = rows
        cm = [["true \\ predicted"] + c["label_names"] + (["no prediction"] if c.get("n_abstained") else [])]
        for i, name in enumerate(c["label_names"]):
            cm.append([name] + c["confusion_matrix"][i] + ([c["abstained_per_class"][i]] if c.get("n_abstained") else []))
        t["confusion_matrix"] = cm
    b = run.get("binary")
    if b:
        t["binary_metrics"] = [["metric", "value"]] + [[k, b[k]] for k in (
            "threshold", "n_samples", "n_waste", "n_non_waste", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1",
            "specificity", "balanced_accuracy", "roc_auc", "waste_acceptance_rate", "non_waste_rejection_rate", "overall_acceptance_rate")]
    d = run.get("detection")
    if d and d["class_aware"].get("available"):
        rows = [["scope", "class", "instances", "precision", "recall", "mAP50", "mAP50-95"]]
        ca, ag = d["class_aware"], d["class_agnostic"]
        rows.append(["class-aware", "all", d["n_ground_truth_boxes"], ca["precision"], ca["recall"], ca["map50"], ca["map50_95"]])
        for r in ca["per_class"]:
            rows.append(["class-aware", r["name"], r["instances"], r["precision"], r["recall"], r["map50"], r["map50_95"]])
        rows.append(["class-agnostic", "all", d["n_ground_truth_boxes"], ag.get("precision"), ag.get("recall"), ag.get("map50"), ag.get("map50_95")])
        t["detection_metrics"] = rows
    inf = run.get("inference")
    if inf and inf.get("stages"):
        rows = [["stage", "n", "mean_ms", "median_ms", "min_ms", "max_ms", "p95_ms"]]
        for k in STAGES:
            s = inf["stages"].get(k)
            if s:
                rows.append([s["label"], s["n"], s["mean"], s["median"], s["min"], s["max"], s["p95"]])
        e = inf.get("end_to_end")
        if e:
            rows.append(["end-to-end (measured)", e["n"], e["mean"], e["median"], e["min"], e["max"], e["p95"]])
        rows.append(["fps", "", inf.get("fps"), "", "", "", ""])
        t["inference_timing"] = rows
    if run.get("per_image"):
        keys = [k for k in run["per_image"][0].keys() if k != "timing_ms"]
        t["per_image"] = [keys] + [[r.get(k) if not isinstance(r.get(k), list) else ";".join(map(str, r.get(k))) for k in keys] for r in run["per_image"]]
    return t


def to_csv(run: dict, table: str = "all") -> tuple[str, str]:
    tables = csv_tables(run)
    buf = io.StringIO()
    w = csv.writer(buf)
    if table != "all":
        if table not in tables:
            raise KeyError(table)
        for row in tables[table]:
            w.writerow(row)
        return buf.getvalue(), f"ecosortai_{run['run_id']}_{table}.csv"
    w.writerow(["EcoSortAI evaluation", run["run_id"], run["task_label"], run["created_at"], run["dataset"].get("display_name") or run["dataset"]["name"]])
    for name, rows in tables.items():
        if name == "per_image":
            continue
        w.writerow([])
        w.writerow([f"# {name}"])
        for row in rows:
            w.writerow(row)
    return buf.getvalue(), f"ecosortai_{run['run_id']}_metrics.csv"


# --------------------------------------------------------------------------- PDF
def to_pdf(run: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    green = colors.HexColor("#143322")
    sage = colors.HexColor("#b9d3a8")
    H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=19, textColor=green, alignment=TA_LEFT, spaceAfter=4)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=green, spaceBefore=10, spaceAfter=4)
    P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=8.8, leading=11.5)
    S = ParagraphStyle("s", parent=P, fontSize=7.6, leading=9.5, textColor=colors.HexColor("#4a5a50"))

    def table(rows, widths=None, header=True, align_right_from=1):
        rows = [[Paragraph(str(c), S) if isinstance(c, str) and len(c) > 40 else ("—" if c is None else c) for c in r] for r in rows]
        tb = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        st = [
            ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d3cc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), green), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8)]
        tb.setStyle(TableStyle(st))
        return tb

    def kv(pairs):
        return table([[k, v] for k, v in pairs], widths=[55 * mm, 120 * mm], header=False, align_right_from=99)

    els = []
    ds = run["dataset"]
    cfg = run["config"]
    els.append(Paragraph("EcoSortAI — Performance Evaluation Report", H1))
    els.append(Paragraph(f"{run['task_label']} · run <b>{run['run_id']}</b> · {run['created_at']} (UTC)", P))
    els.append(Spacer(1, 4))
    els.append(Paragraph("All figures in this report were computed by the EcoSortAI backend from the model predictions on the "
                         "dataset listed below. Historical checkpoint metrics are not mixed in.", S))

    els.append(Paragraph("Model configuration", H2))
    mods = cfg["models"]
    def ck(m):
        c = (m or {}).get("checkpoint") or {}
        return f"{c.get('file', '—')} · sha256 {c.get('sha256', '—')}" if c else "—"
    els.append(kv([
        ("Seven-class classifier", f"{mods['classifier'].get('architecture')} · {ck(mods['classifier'])}"),
        ("Binary waste filter", f"{mods['binary_filter'].get('architecture') or 'not loaded'} · {ck(mods['binary_filter'])}"),
        ("YOLOv8 detector", f"{mods['detector'].get('name') or 'not loaded'} · {ck(mods['detector'])}"),
        ("Binary threshold", f"P(waste) ≥ {mods['binary_filter']['threshold']}"),
        ("YOLO thresholds (inference)", f"conf {mods['detector']['inference_confidence']}, NMS IoU {mods['detector']['inference_iou']}, agnostic NMS"),
        ("Evaluation parameters", ", ".join(f"{k}={v}" for k, v in cfg["params"].items())),
        ("Device", mods.get("device")),
    ]))

    els.append(Paragraph("Dataset", H2))
    dist = ds.get("class_distribution") or {}
    els.append(kv([
        ("Name", ds.get("display_name") or ds["name"]),
        ("Source / label format", f"{ds.get('source')} · {ds.get('label_format')}"),
        ("Images found / evaluated", f"{ds.get('n_images_found')} / {ds.get('n_images_evaluated')}"),
        ("Invalid images", ds.get("n_invalid", 0)),
        ("Class distribution", ", ".join(f"{k}: {v}" for k, v in dist.items()) or "—"),
        ("Supervised metrics", "yes" if run.get("supervised") else "no — ground truth not available"),
    ]))

    c = run.get("classification")
    if c:
        els.append(Paragraph("Classification performance", H2))
        els.append(Paragraph(f"Mode: <b>{c['mode']}</b>. {c['prediction_rule']}", S))
        els.append(Spacer(1, 3))
        els.append(kv([
            ("Accuracy", f"{_pct(c['accuracy'])} ({c['n_correct']}/{c['n_samples']})"),
            ("Macro avg P / R / F1", " / ".join(_num(c["macro_avg"][k]) for k in ("precision", "recall", "f1"))),
            ("Weighted avg P / R / F1", " / ".join(_num(c["weighted_avg"][k]) for k in ("precision", "recall", "f1"))),
            ("No prediction", f"{c['n_abstained']} (coverage {_pct(c['coverage'])})"),
        ]))
        els.append(Spacer(1, 5))
        rows = [["Class", "Precision", "Recall", "F1-score", "Support"]]
        for r in c["per_class"]:
            rows.append([r["label"], _num(r["precision"]), _num(r["recall"]), _num(r["f1"]), r["support"]])
        rows += [["Macro avg", *(_num(c["macro_avg"][k]) for k in ("precision", "recall", "f1")), c["n_samples"]],
                 ["Weighted avg", *(_num(c["weighted_avg"][k]) for k in ("precision", "recall", "f1")), c["n_samples"]]]
        els.append(table(rows, widths=[50 * mm, 28 * mm, 28 * mm, 28 * mm, 25 * mm]))

        # confusion matrix heat table
        names = c["label_names"]
        cm = c["confusion_matrix"]
        extra = bool(c.get("n_abstained"))
        hdr = ["True \\ Pred"] + [n.replace("Paper & Cardboard", "Paper") for n in names] + (["None"] if extra else [])
        rows = [hdr]
        for i, n in enumerate(names):
            rows.append([n.replace("Paper & Cardboard", "Paper")] + cm[i] + ([c["abstained_per_class"][i]] if extra else []))
        mx = max([max(r) for r in cm] + [1])
        w = [26 * mm] + [19 * mm] * (len(hdr) - 1)
        tb = table(rows, widths=w)
        st = []
        lo, hi = (0.96, 0.97, 0.95), (0.13, 0.36, 0.22)  # paper → forest green
        for i in range(len(names)):
            for j in range(len(names)):
                a = cm[i][j] / mx
                rgb = [lo[k] + (hi[k] - lo[k]) * a for k in range(3)]
                st.append(("BACKGROUND", (j + 1, i + 1), (j + 1, i + 1), colors.Color(*rgb)))
                if a > 0.55:
                    st.append(("TEXTCOLOR", (j + 1, i + 1), (j + 1, i + 1), colors.white))
        tb.setStyle(TableStyle(st))
        els.append(Spacer(1, 6))
        els.append(KeepTogether([Paragraph("Confusion matrix (counts; rows = ground truth)", P), Spacer(1, 2), tb]))

    b = run.get("binary")
    if b:
        els.append(Paragraph("Binary waste verification", H2))
        els.append(Paragraph(run.get("binary_input", ""), S))
        els.append(kv([
            ("Samples (waste / non-waste)", f"{b['n_samples']} ({b['n_waste']} / {b['n_non_waste']})"),
            ("Accuracy", _pct(b["accuracy"])),
            ("Precision / Recall / F1 (waste = positive)", f"{_num(b['precision'])} / {_num(b['recall'])} / {_num(b['f1'])}"),
            ("Balanced accuracy / ROC-AUC", f"{_num(b['balanced_accuracy'])} / {_num(b['roc_auc'])}"),
            ("Waste acceptance rate (TP / all true waste)", _pct(b["waste_acceptance_rate"])),
            ("Non-waste rejection rate (TN / all true non-waste)", _pct(b["non_waste_rejection_rate"])),
            ("Confusion [TN FP; FN TP]", f"[{b['tn']} {b['fp']}; {b['fn']} {b['tp']}]"),
        ]))
    elif run.get("binary_unsupervised"):
        u = run["binary_unsupervised"]
        els.append(Paragraph("Binary waste verification (unlabelled)", H2))
        els.append(Paragraph(f"Accepted {u['accepted']} of {u['n_samples']} images ({_pct(u['overall_acceptance_rate'])}). {u['note']}", P))

    d = run.get("detection")
    if d and d["class_aware"].get("available"):
        ca, ag = d["class_aware"], d["class_agnostic"]
        els.append(Paragraph("YOLOv8 object detection", H2))
        els.append(Paragraph(d["methodology"], S))
        els.append(Spacer(1, 3))
        rows = [["Scope", "Instances", "Precision", "Recall", "mAP@0.50", "mAP@0.50:0.95"],
                ["Class-aware (all)", d["n_ground_truth_boxes"], _num(ca["precision"]), _num(ca["recall"]), _num(ca["map50"]), _num(ca["map50_95"])]]
        for r in ca["per_class"]:
            rows.append([f"  {r['name']}", r["instances"], _num(r["precision"]), _num(r["recall"]), _num(r["map50"]), _num(r["map50_95"])])
        rows.append(["Class-agnostic", d["n_ground_truth_boxes"], _num(ag.get("precision")), _num(ag.get("recall")), _num(ag.get("map50")), _num(ag.get("map50_95"))])
        els.append(table(rows, widths=[48 * mm, 22 * mm, 24 * mm, 24 * mm, 26 * mm, 30 * mm]))
        els.append(Paragraph(f"conf ≥ {d['confidence_threshold']}, NMS IoU {d['nms_iou_threshold']}, max_det {d['max_det']}; "
                             f"{d['n_images']} images, {d['n_predicted_boxes']} predicted boxes.", S))

    inf = run.get("inference")
    if inf and inf.get("stages"):
        els.append(Paragraph("Inference performance", H2))
        inc = inf.get("includes", {})
        incl = ", ".join(k.replace("_", " ") for k, v in inc.items() if v) or "—"
        excl = ", ".join(k.replace("_", " ") for k, v in inc.items() if not v) or "—"
        els.append(Paragraph(f"Measured on <b>{inf['device']}</b>, batch size 1, {inf['n_images_timed']} images after warm-up. "
                             f"Timed: {incl}. Not timed: {excl}. {inf['fps_definition']}", S))
        rows = [["Stage", "Mean", "Median", "Min", "Max", "P95"]]
        for k in STAGES:
            s = inf["stages"].get(k)
            if s:
                rows.append([s["label"], _ms(s["mean"]), _ms(s["median"]), _ms(s["min"]), _ms(s["max"]), _ms(s["p95"])])
        e = inf.get("end_to_end")
        if e:
            rows.append(["End-to-end (measured wall time)", _ms(e["mean"]), _ms(e["median"]), _ms(e["min"]), _ms(e["max"]), _ms(e["p95"])])
        els.append(table(rows, widths=[66 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm]))
        els.append(Paragraph(f"Throughput: <b>{_num(inf.get('fps'), 2)} FPS</b> (images per second).", P))

    obs = observations(run)
    if obs:
        els.append(Paragraph("Key observations", H2))
        for o in obs:
            els.append(Paragraph(f"• {o}", P))
    els.append(Paragraph("Limitations and evaluation notes", H2))
    for o in limitations(run):
        els.append(Paragraph(f"• {o}", P))

    buf = io.BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b7a70"))
        canvas.drawString(18 * mm, 10 * mm, f"EcoSortAI evaluation {run['run_id']} · generated from measured predictions")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(sage)
        canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"EcoSortAI evaluation {run['run_id']}", author="EcoSortAI")
    doc.build(els, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
