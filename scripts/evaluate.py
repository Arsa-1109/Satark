"""Evaluate Full ONNX Hybrid & Lite Fallback across G1-G6 held-out gates."""

from __future__ import annotations

import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from satark.engine import CATEGORIES, SatarkEngine  # noqa: E402

DATA_PROC = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\data\processed")
BENCH_FULL_DIR = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\benchmarks")
LITE_SUMMARY_PATH = ROOT / "models" / "lite" / "benchmarks_summary.json"
PWA_SUMMARY_PATH = ROOT / "frontend" / "public" / "benchmarks_summary.json"


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    curve: list[dict] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            curve.append({"bin": i, "lo": round(lo, 2), "hi": round(hi, 2), "count": 0, "avg_conf": 0.0, "avg_acc": 0.0})
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (cnt / n) * abs(avg_acc - avg_conf)
        curve.append({
            "bin": i,
            "lo": round(lo, 2),
            "hi": round(hi, 2),
            "count": cnt,
            "avg_conf": round(avg_conf, 4),
            "avg_acc": round(avg_acc, 4),
        })
    return float(ece), curve


def evaluate_mode(
    engine: SatarkEngine,
    unseen_df: pd.DataFrame,
    hn_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    force_lite: bool,
) -> tuple[dict, list[dict]]:
    # 1. Unseen test
    y_true = unseen_df["label"].to_numpy(dtype=np.int32)
    y_pred: list[int] = []
    y_prob: list[float] = []
    cat_true: list[str] = []
    cat_pred: list[str] = []

    latencies: list[float] = []
    for idx, row in unseen_df.iterrows():
        t0 = time.perf_counter()
        res = engine.analyze(str(row["text"]), force_lite=force_lite)
        dt = (time.perf_counter() - t0) * 1000.0
        if len(latencies) < 120:
            latencies.append(dt)
        pred_lbl = 1 if res["verdict"] == "SCAM" else 0
        y_pred.append(pred_lbl)
        y_prob.append(float(res["scam_probability"]))

        cat = str(row["category"]) if pd.notna(row["category"]) else ""
        if row["label"] == 1 and cat in CATEGORIES:
            cat_true.append(cat)
            cat_pred.append(res["category"] if res["category"] in CATEGORIES else "Other")

    y_pred_arr = np.array(y_pred, dtype=np.int32)
    y_prob_arr = np.array(y_prob, dtype=np.float64)

    unseen_f1 = float(f1_score(y_true, y_pred_arr))
    macro_f1 = float(f1_score(cat_true, cat_pred, labels=list(CATEGORIES), average="macro"))
    ece, curve = compute_ece(y_true, y_prob_arr, n_bins=10)
    brier = float(brier_score_loss(y_true, y_prob_arr))

    # 2. Hard-Negative FPR
    hn_fp = 0
    for txt in hn_df["text"].astype(str).tolist():
        res = engine.analyze(txt, force_lite=force_lite)
        if res["verdict"] == "SCAM":
            hn_fp += 1
    hn_fpr_pct = (hn_fp / max(len(hn_df), 1)) * 100.0

    # 3. Adversarial Recall
    adv_tp = 0
    for txt in adv_df["text"].astype(str).tolist():
        res = engine.analyze(txt, force_lite=force_lite)
        if res["verdict"] == "SCAM":
            adv_tp += 1
    adv_recall = adv_tp / max(len(adv_df), 1)

    median_ms = float(statistics.median(latencies[5:])) if len(latencies) > 10 else float(statistics.median(latencies))

    metrics = {
        "unseen_f1": round(unseen_f1, 4),
        "hard_neg_fpr_pct": round(hn_fpr_pct, 2),
        "hard_neg_fp_count": hn_fp,
        "hard_neg_total": len(hn_df),
        "macro_f1_5class": round(macro_f1, 4),
        "adversarial_recall": round(adv_recall, 4),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "median_latency_ms": round(median_ms, 2),
    }
    return metrics, curve


def main() -> None:
    BENCH_FULL_DIR.mkdir(parents=True, exist_ok=True)
    unseen_df = pd.read_csv(DATA_PROC / "test_unseen.csv")
    hn_df = pd.read_csv(DATA_PROC / "test_hard_negatives.csv")
    adv_df = pd.read_csv(DATA_PROC / "test_adversarial.csv")

    engine = SatarkEngine()
    print("Evaluating Full ONNX Hybrid...")
    full_metrics, full_curve = evaluate_mode(engine, unseen_df, hn_df, adv_df, force_lite=False)
    print("Evaluating Lite Fallback...")
    lite_metrics, lite_curve = evaluate_mode(engine, unseen_df, hn_df, adv_df, force_lite=True)

    summary = {
        "full_onnx_hybrid": full_metrics,
        "lite_fallback": lite_metrics,
        "gates": {
            "G1_unseen_f1_ge_0.94": full_metrics["unseen_f1"] >= 0.94,
            "G2_hard_neg_fpr_le_2pct": full_metrics["hard_neg_fpr_pct"] <= 2.0,
            "G3_macro_f1_ge_0.88": full_metrics["macro_f1_5class"] >= 0.88,
            "G4_adv_recall_ge_0.90": full_metrics["adversarial_recall"] >= 0.90,
            "G5_ece_le_0.06_brier_le_0.08": full_metrics["ece"] <= 0.06 and full_metrics["brier_score"] <= 0.08,
            "G6_latency_lt_80ms_full_10ms_lite": full_metrics["median_latency_ms"] < 80.0 and lite_metrics["median_latency_ms"] < 10.0,
        },
        "calibration_curve": full_curve,
    }

    (BENCH_FULL_DIR / "metrics_full.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(full_curve).to_csv(BENCH_FULL_DIR / "calibration_curve.csv", index=False)

    LITE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    PWA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LITE_SUMMARY_PATH, PWA_SUMMARY_PATH)

    print("\n=== FULL ONNX HYBRID METRICS ===")
    print(json.dumps(full_metrics, indent=2))
    print("\n=== LITE FALLBACK METRICS ===")
    print(json.dumps(lite_metrics, indent=2))
    print("\n=== GATES G1-G6 ===")
    print(json.dumps(summary["gates"], indent=2))

    if not all(summary["gates"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
