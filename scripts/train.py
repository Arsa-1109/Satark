"""Train Lite Fallback (<4 MB) and Full ONNX Hybrid models + Platt Calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_pwa import export_bundle  # noqa: E402
from satark.backbone import OnnxBackbone  # noqa: E402
from satark.engine import CATEGORIES  # noqa: E402
from satark.forensics import extract_forensics  # noqa: E402
from satark.normalizer import normalize_text  # noqa: E402

DATA_PROC = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\data\processed")
LITE_DIR = ROOT / "models" / "lite"
ONNX_DIR = ROOT / "models" / "onnx"


def _prepare_texts(df: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    cleans: list[str] = []
    canons: list[str] = []
    feats: list[list[float]] = []
    for txt in df["text"].astype(str).tolist():
        norm = normalize_text(txt, validate=False)
        fr = extract_forensics(norm)
        cleans.append(norm.clean_text)
        canons.append(norm.canonical_text)
        feats.append(fr.features)
    return cleans, canons, np.array(feats, dtype=np.float32)


def _fit_platt(logits: np.ndarray, y_true: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    platt = LogisticRegression(C=1e6, max_iter=500)
    platt.fit(logits.reshape(-1, 1), y_true, sample_weight=weights)
    return float(platt.coef_[0][0]), float(platt.intercept_[0])


def main() -> None:
    LITE_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_PROC / "train.csv")
    val_df = pd.read_csv(DATA_PROC / "val.csv")

    print(f"Preparing train ({len(train_df)}) and val ({len(val_df)}) features...")
    tr_clean, tr_canon, tr_f = _prepare_texts(train_df)
    va_clean, va_canon, va_f = _prepare_texts(val_df)

    y_tr = train_df["label"].to_numpy(dtype=np.int32)
    w_tr = train_df["sample_weight"].fillna(1.0).to_numpy(dtype=np.float32)
    y_va = val_df["label"].to_numpy(dtype=np.int32)
    w_va = val_df["sample_weight"].fillna(1.0).to_numpy(dtype=np.float32)

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=8000,
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        min_df=2,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    forensic_scaler = StandardScaler()

    xc_tr = char_tfidf.fit_transform(tr_canon)
    xw_tr = word_tfidf.fit_transform(tr_canon)
    xf_tr = forensic_scaler.fit_transform(tr_f).astype(np.float32)
    x_tr_sparse = hstack([xc_tr, xw_tr, csr_matrix(xf_tr)], format="csr")

    xc_va = char_tfidf.transform(va_canon)
    xw_va = word_tfidf.transform(va_canon)
    xf_va = forensic_scaler.transform(va_f).astype(np.float32)
    x_va_sparse = hstack([xc_va, xw_va, csr_matrix(xf_va)], format="csr")

    joblib.dump(
        {"char_tfidf": char_tfidf, "word_tfidf": word_tfidf, "forensic_scaler": forensic_scaler},
        LITE_DIR / "vectorizers.joblib",
        compress=3,
    )

    # --- Lite Stage A ---
    print("Training Lite Stage A...")
    lr_lite_a = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)
    lr_lite_a.fit(x_tr_sparse, y_tr, sample_weight=w_tr)
    va_logits_lite = lr_lite_a.decision_function(x_va_sparse)
    pa_lite, pb_lite = _fit_platt(va_logits_lite, y_va, w_va)
    joblib.dump(
        {"clf": lr_lite_a, "platt_a": pa_lite, "platt_b": pb_lite, "threshold": 0.5},
        LITE_DIR / "stage_a_lite.joblib",
        compress=3,
    )

    # --- Lite Stage B (5-Class) ---
    print("Training Lite Stage B...")
    cat_mask_tr = (train_df["label"] == 1) & (train_df["category"].fillna("").isin(CATEGORIES))
    x_tr_cat_sparse = x_tr_sparse[cat_mask_tr.to_numpy()]
    y_tr_cat = train_df.loc[cat_mask_tr, "category"].to_numpy()
    lr_lite_b = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)
    lr_lite_b.fit(x_tr_cat_sparse, y_tr_cat)
    joblib.dump(
        {"clf": lr_lite_b, "classes": list(lr_lite_b.classes_)},
        LITE_DIR / "stage_b_lite.joblib",
        compress=3,
    )

    lite_bytes = sum((LITE_DIR / f).stat().st_size for f in ("vectorizers.joblib", "stage_a_lite.joblib", "stage_b_lite.joblib"))
    print(f"Lite artefacts total size: {lite_bytes / (1024 * 1024):.2f} MB")

    # Export PWA bundle
    export_bundle()

    # --- Full ONNX Hybrid ---
    print("Encoding train & val with Frozen MiniLM INT8 ONNX...")
    backbone = OnnxBackbone(ONNX_DIR, max_length=128)
    tr_embs = backbone.encode(tr_clean, batch_size=64)
    va_embs = backbone.encode(va_clean, batch_size=64)

    x_tr_hybrid = hstack([csr_matrix(tr_embs), x_tr_sparse], format="csr")
    x_va_hybrid = hstack([csr_matrix(va_embs), x_va_sparse], format="csr")

    print("Training Full ONNX Hybrid Stage A (LR + LightGBM stacking)...")
    lr_full_a = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)
    oof_lr_logits = cross_val_predict(
        LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000),
        x_tr_hybrid,
        y_tr,
        cv=5,
        method="decision_function",
    )
    lr_full_a.fit(x_tr_hybrid, y_tr, sample_weight=w_tr)

    lgbm_tr_in = np.hstack([tr_embs, xf_tr, oof_lr_logits[:, None]])
    lgbm_full_a = LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )
    lgbm_full_a.fit(lgbm_tr_in, y_tr, sample_weight=w_tr)

    va_lr_logits = lr_full_a.decision_function(x_va_hybrid)
    va_lgbm_in = np.hstack([va_embs, xf_va, va_lr_logits[:, None]])
    va_lgbm_probs = np.clip(lgbm_full_a.predict_proba(va_lgbm_in)[:, 1], 1e-6, 1.0 - 1e-6)
    va_lgbm_logits = np.log(va_lgbm_probs / (1.0 - va_lgbm_probs))

    alpha = 0.5
    va_blended_logits = alpha * va_lr_logits + (1.0 - alpha) * va_lgbm_logits
    pa_full, pb_full = _fit_platt(va_blended_logits, y_va, w_va)

    joblib.dump(
        {"lr": lr_full_a, "lgbm": lgbm_full_a, "platt_a": pa_full, "platt_b": pb_full, "alpha": alpha, "threshold": 0.5},
        ONNX_DIR / "stage_a_full.joblib",
        compress=3,
    )

    print("Training Full ONNX Hybrid Stage B...")
    x_tr_cat_hybrid = x_tr_hybrid[cat_mask_tr.to_numpy()]
    lr_full_b = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)
    lr_full_b.fit(x_tr_cat_hybrid, y_tr_cat)
    joblib.dump(
        {"clf": lr_full_b, "classes": list(lr_full_b.classes_)},
        ONNX_DIR / "stage_b_full.joblib",
        compress=3,
    )
    print("Training complete!")


if __name__ == "__main__":
    main()
