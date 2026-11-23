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
    stress_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    multi_df: pd.DataFrame,
    ads_df: pd.DataFrame,
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

    # 4. Held-out Real-World Stress Suite (Gate G10)
    rw_correct = 0
    rw_safe_fp = 0
    rw_scam_fn = 0
    for _, row in stress_df.iterrows():
        res = engine.analyze(str(row["text"]), force_lite=force_lite)
        pred = 1 if res["verdict"] == "SCAM" else 0
        lbl = int(row["label"])
        if pred == lbl:
            rw_correct += 1
        elif lbl == 0 and pred == 1:
            rw_safe_fp += 1
        elif lbl == 1 and pred == 0:
            rw_scam_fn += 1
    rw_acc = rw_correct / max(len(stress_df), 1)

    # 5. Strictly Blind OOD SAFE Suite (Gate G11)
    ood_fp = 0
    for txt in ood_df["text"].astype(str).tolist():
        res = engine.analyze(txt, force_lite=force_lite)
        if res["verdict"] == "SCAM":
            ood_fp += 1
    ood_fpr_pct = (ood_fp / max(len(ood_df), 1)) * 100.0

    # 6. Headerless Transactional & Promotional SAFE Suite (Gate G12)
    tx_fp = 0
    for txt in tx_df["text"].astype(str).tolist():
        res = engine.analyze(txt, force_lite=force_lite)
        if res["verdict"] == "SCAM":
            tx_fp += 1
    tx_fpr_pct = (tx_fp / max(len(tx_df), 1)) * 100.0

    # 7. Blind Multilingual + Novel Scam + Tricky Safe Suite (Gate G13)
    multi_ok = 0
    for _, row in multi_df.iterrows():
        res = engine.analyze(str(row["text"]), force_lite=force_lite)
        pred = 1 if res["verdict"] == "SCAM" else 0
        if pred == int(row["label"]):
            multi_ok += 1
    multi_acc = multi_ok / max(len(multi_df), 1)

    # 8. Blind Commercial Ads SAFE Suite (Gate G14)
    ads_fp = 0
    for txt in ads_df["text"].astype(str).tolist():
        res = engine.analyze(txt, force_lite=force_lite)
        if res["verdict"] == "SCAM":
            ads_fp += 1
    ads_fpr_pct = (ads_fp / max(len(ads_df), 1)) * 100.0

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
        "realworld_accuracy": round(rw_acc, 4),
        "realworld_safe_fp": rw_safe_fp,
        "realworld_scam_fn": rw_scam_fn,
        "realworld_total": len(stress_df),
        "ood_safe_fpr_pct": round(ood_fpr_pct, 2),
        "ood_safe_fp_count": ood_fp,
        "ood_safe_total": len(ood_df),
        "transactional_fpr_pct": round(tx_fpr_pct, 2),
        "transactional_fp_count": tx_fp,
        "transactional_total": len(tx_df),
        "blind_multilingual_accuracy": round(multi_acc, 4),
        "blind_multilingual_correct": multi_ok,
        "blind_multilingual_total": len(multi_df),
        "ads_safe_fpr_pct": round(ads_fpr_pct, 2),
        "ads_safe_fp_count": ads_fp,
        "ads_safe_total": len(ads_df),
    }
    return metrics, curve


def main() -> None:
    BENCH_FULL_DIR.mkdir(parents=True, exist_ok=True)
    unseen_df = pd.read_csv(DATA_PROC / "test_unseen.csv")
    hn_df = pd.read_csv(DATA_PROC / "test_hard_negatives.csv")
    adv_df = pd.read_csv(DATA_PROC / "test_adversarial.csv")
    stress_df = pd.read_csv(DATA_PROC / "test_realworld_stress.csv")
    ood_df = pd.read_csv(DATA_PROC / "test_ood_safe.csv")
    tx_df = pd.read_csv(DATA_PROC / "test_transactional_safe.csv")
    multi_df = pd.read_csv(DATA_PROC / "test_blind_multilingual.csv")
    ads_df = pd.read_csv(DATA_PROC / "test_ads_safe.csv")

    engine = SatarkEngine()
    print("Evaluating Full ONNX Hybrid...")
    full_metrics, full_curve = evaluate_mode(engine, unseen_df, hn_df, adv_df, stress_df, ood_df, tx_df, multi_df, ads_df, force_lite=False)
    print("Evaluating Lite Fallback...")
    lite_metrics, lite_curve = evaluate_mode(engine, unseen_df, hn_df, adv_df, stress_df, ood_df, tx_df, multi_df, ads_df, force_lite=True)

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
            "G10_realworld_stress_ge_92pct": full_metrics["realworld_accuracy"] >= 0.92 and full_metrics["realworld_safe_fp"] <= 1,
            "G11_ood_safe_fpr_le_5pct": full_metrics["ood_safe_fpr_pct"] <= 5.0 and lite_metrics["ood_safe_fpr_pct"] <= 5.0,
            "G12_transactional_promo_fpr_le_4pct": full_metrics["transactional_fpr_pct"] <= 4.0 and lite_metrics["transactional_fpr_pct"] <= 4.0,
            "G13_blind_multilingual_ge_92pct": full_metrics["blind_multilingual_accuracy"] >= 0.92,
            "G14_commercial_ads_fpr_le_5pct": full_metrics["ads_safe_fpr_pct"] <= 5.0 and lite_metrics["ads_safe_fpr_pct"] <= 5.0,
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
    print("\n=== GATES G1-G11 ===")
    print(json.dumps(summary["gates"], indent=2))

    samples_path = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\samples.txt")
    if samples_path.exists():
        print("\n=== USER SAMPLES.TXT AUDIT ===")
        for idx, msg in enumerate([m.strip() for m in samples_path.read_text(encoding="utf-8").split("\n\n") if m.strip()], 1):
            rf = engine.analyze(msg, force_lite=False)
            rl = engine.analyze(msg, force_lite=True)
            print(f"  Sample {idx}: Full={rf['verdict']} (p={rf['scam_probability']}) | Lite={rl['verdict']} (p={rl['scam_probability']}) | {msg[:75]}...")

    if not all(summary["gates"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
