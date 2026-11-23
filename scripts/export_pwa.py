"""Export Lite Fallback weights to frontend/public/model_bundle.js (<2 MB) for Offline PWA."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LITE_DIR = ROOT / "models" / "lite"
OUT_JS = ROOT / "frontend" / "public" / "model_bundle.js"


def _round_list(arr: np.ndarray, decimals: int = 4) -> list[float]:
    return [round(float(x), decimals) for x in arr.tolist()]


def export_bundle() -> Path:
    vec_bundle = joblib.load(LITE_DIR / "vectorizers.joblib")
    stage_a = joblib.load(LITE_DIR / "stage_a_lite.joblib")
    stage_b = joblib.load(LITE_DIR / "stage_b_lite.joblib")

    char_vec = vec_bundle["char_tfidf"]
    word_vec = vec_bundle["word_tfidf"]
    scaler = vec_bundle["forensic_scaler"]

    # Order vocabularies by feature index 0..N-1
    char_vocab = [tok for tok, _ in sorted(char_vec.vocabulary_.items(), key=lambda kv: kv[1])]
    word_vocab = [tok for tok, _ in sorted(word_vec.vocabulary_.items(), key=lambda kv: kv[1])]

    clf_a = stage_a["clf"]
    clf_b = stage_b["clf"]

    bundle = {
        "version": "1.0.0",
        "char_vocab": char_vocab,
        "char_idf": _round_list(char_vec.idf_, 4),
        "word_vocab": word_vocab,
        "word_idf": _round_list(word_vec.idf_, 4),
        "scaler_mean": _round_list(scaler.mean_, 5),
        "scaler_scale": _round_list(scaler.scale_, 5),
        "stage_a": {
            "coef": _round_list(clf_a.coef_[0], 4),
            "intercept": round(float(clf_a.intercept_[0]), 5),
            "platt_a": round(float(stage_a["platt_a"]), 5),
            "platt_b": round(float(stage_a["platt_b"]), 5),
            "threshold": round(float(stage_a["threshold"]), 4),
        },
        "stage_b": {
            "classes": [str(c) for c in stage_b["classes"]],
            "coef": [_round_list(row, 4) for row in clf_b.coef_],
            "intercept": _round_list(clf_b.intercept_, 5),
        },
    }

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    payload = "window.__SATARK_MODEL__ = " + json.dumps(bundle, separators=(",", ":"), ensure_ascii=True) + ";\n"
    OUT_JS.write_text(payload, encoding="utf-8")
    size_mb = OUT_JS.stat().st_size / (1024 * 1024)
    print(f"Exported PWA model bundle to {OUT_JS} ({size_mb:.2f} MB)")
    return OUT_JS


if __name__ == "__main__":
    export_bundle()
