"""Train Lite Fallback (<4 MB) and Full ONNX Hybrid models + Platt Calibration."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_pwa import export_bundle  # noqa: E402
from satark.backbone import OnnxBackbone  # noqa: E402
from satark.engine import CATEGORIES  # noqa: E402
from satark.forensics import FEATURE_NAMES, extract_forensics  # noqa: E402
from satark.normalizer import normalize_text  # noqa: E402


def _resolve_data_proc() -> Path:
    env_dir = os.environ.get("SATARK_DATA_DIR")
    if env_dir:
        return Path(env_dir) / "processed"
    default_dir = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\data\processed")
    if default_dir.parent.parent.exists():
        return default_dir
    return ROOT / "data" / "processed"


DATA_PROC = _resolve_data_proc()
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
    platt = LogisticRegression(C=10.0, max_iter=500)
    platt.fit(logits.reshape(-1, 1), y_true, sample_weight=weights)
    return float(platt.coef_[0][0]), float(platt.intercept_[0])


def _select_threshold(probs: np.ndarray, y_true: np.ndarray, max_fpr: float = 0.015) -> float:
    """Pick highest-F1 threshold on val.csv within a fair FPR cap."""
    best_t = 0.50
    best_f1 = -1.0
    neg_mask = y_true == 0
    for t in np.arange(0.42, 0.63, 0.02):
        preds = (probs >= t).astype(np.int32)
        fpr = float((preds[neg_mask] == 1).mean()) if neg_mask.any() else 0.0
        if fpr <= max_fpr:
            f1 = float(f1_score(y_true, preds))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))
    if best_f1 < 0:
        for t in np.arange(0.42, 0.63, 0.02):
            preds = (probs >= t).astype(np.int32)
            f1 = float(f1_score(y_true, preds))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))
    return best_t


def _encode_with_cache(backbone: OnnxBackbone, texts: list[str], cache_path: Path) -> np.ndarray:
    """Cache MiniLM embeddings by text SHA-256 so future retrains run in <20s."""
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        for k, v in zip(data["keys"], data["embs"]):
            cache[str(k)] = v

    keys = [hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()[:24] for t in texts]
    missing_indices = [i for i, k in enumerate(keys) if k not in cache]
    if missing_indices:
        # Deduplicate missing texts before encoding
        uniq_missing: dict[str, str] = {}
        for idx in missing_indices:
            uniq_missing[keys[idx]] = texts[idx]
        m_keys = list(uniq_missing.keys())
        m_texts = [uniq_missing[k] for k in m_keys]
        print(f"Encoding {len(m_texts)} uncached texts via MiniLM INT8 ONNX...")
        m_embs = backbone.encode(m_texts, batch_size=64)
        for k, emb in zip(m_keys, m_embs):
            cache[k] = emb
        np.savez_compressed(
            cache_path,
            keys=np.array(list(cache.keys())),
            embs=np.vstack(list(cache.values())).astype(np.float32),
        )
    else:
        print(f"Loaded all {len(texts)} embeddings from cache ({cache_path.name}).")

    return np.vstack([cache[k] for k in keys]).astype(np.float32)


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

    # --- True sample-weight-aware class balance (no double amplification) ---
    pos_w_sum = float(w_tr[y_tr == 1].sum())
    neg_w_sum = float(w_tr[y_tr == 0].sum())
    target_pos_ratio = 0.72
    scale_pos = (target_pos_ratio * neg_w_sum) / max(pos_w_sum, 1e-6)
    w_tr_bal = np.where(y_tr == 1, w_tr * scale_pos, w_tr).astype(np.float32)
    print(f"Effective train weights: neg={neg_w_sum:.1f}, pos_raw={pos_w_sum:.1f}, scale_pos={scale_pos:.3f}, pos_bal={float(w_tr_bal[y_tr==1].sum()):.1f}")

    # --- Lite Stage A ---
    print("Training Lite Stage A (sample-weight balanced)...")
    stage_a_cw = None
    lr_lite_a = LogisticRegression(C=1.5, class_weight=stage_a_cw, max_iter=1000)
    lr_lite_a.fit(x_tr_sparse, y_tr, sample_weight=w_tr_bal)
    va_logits_lite = lr_lite_a.decision_function(x_va_sparse)
    pa_lite, pb_lite = _fit_platt(va_logits_lite, y_va, w_va)
    va_probs_lite = 1.0 / (1.0 + np.exp(-(pa_lite * va_logits_lite + pb_lite)))
    th_lite = _select_threshold(va_probs_lite, y_va, max_fpr=0.015)
    print(f"Selected Lite Stage A threshold={th_lite} (platt_a={pa_lite:.4f}, platt_b={pb_lite:.4f}, intercept={lr_lite_a.intercept_[0]:.4f})")
    joblib.dump(
        {"clf": lr_lite_a, "platt_a": pa_lite, "platt_b": pb_lite, "threshold": th_lite},
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
    print("Encoding train & val with Frozen MiniLM INT8 ONNX (cached)...")
    backbone = OnnxBackbone(ONNX_DIR, max_length=128)
    cache_path = DATA_PROC / "minilm_emb_cache.npz"
    tr_embs = _encode_with_cache(backbone, tr_clean, cache_path)
    va_embs = _encode_with_cache(backbone, va_clean, cache_path)

    x_tr_hybrid = hstack([csr_matrix(tr_embs), x_tr_sparse], format="csr")
    x_va_hybrid = hstack([csr_matrix(va_embs), x_va_sparse], format="csr")

    print("Training Full ONNX Hybrid Stage A (LR + LightGBM stacking, sample-weight balanced)...")
    lr_full_a = LogisticRegression(C=1.5, class_weight=stage_a_cw, max_iter=1000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lr_logits = np.zeros(len(y_tr), dtype=np.float64)
    for tr_i, va_i in skf.split(x_tr_hybrid, y_tr):
        lr_fold = LogisticRegression(C=1.5, class_weight=stage_a_cw, max_iter=1000)
        lr_fold.fit(x_tr_hybrid[tr_i], y_tr[tr_i], sample_weight=w_tr_bal[tr_i])
        oof_lr_logits[va_i] = lr_fold.decision_function(x_tr_hybrid[va_i])
    lr_full_a.fit(x_tr_hybrid, y_tr, sample_weight=w_tr_bal)

    lgbm_tr_in = np.hstack([tr_embs, xf_tr, oof_lr_logits[:, None]])
    lgbm_full_a = LGBMClassifier(
        n_estimators=180,
        num_leaves=23,
        min_child_samples=25,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        class_weight=stage_a_cw,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )
    lgbm_full_a.fit(lgbm_tr_in, y_tr, sample_weight=w_tr_bal)

    va_lr_logits = lr_full_a.decision_function(x_va_hybrid)
    va_lgbm_in = np.hstack([va_embs, xf_va, va_lr_logits[:, None]])
    va_lgbm_probs = np.clip(lgbm_full_a.predict_proba(va_lgbm_in)[:, 1], 1e-6, 1.0 - 1e-6)
    va_lgbm_logits = np.log(va_lgbm_probs / (1.0 - va_lgbm_probs))

    best_alpha, best_loss, pa_full, pb_full, best_probs_full = 0.70, 999.0, 1.0, 0.0, None
    for cand_alpha in (0.60, 0.65, 0.70, 0.75, 0.80):
        blended = cand_alpha * va_lr_logits + (1.0 - cand_alpha) * va_lgbm_logits
        pa, pb = _fit_platt(blended, y_va, w_va)
        probs = 1.0 / (1.0 + np.exp(-(pa * blended + pb)))
        loss = float(log_loss(y_va, probs, sample_weight=w_va))
        if loss < best_loss:
            best_alpha, best_loss, pa_full, pb_full, best_probs_full = cand_alpha, loss, pa, pb, probs
    th_full = _select_threshold(best_probs_full, y_va, max_fpr=0.015)
    print(f"Selected Full Stage A blend alpha={best_alpha}, threshold={th_full} (platt_a={pa_full:.4f}, platt_b={pb_full:.4f}, intercept={lr_full_a.intercept_[0]:.4f}, val log_loss={best_loss:.4f})")

    joblib.dump(
        {"lr": lr_full_a, "lgbm": lgbm_full_a, "platt_a": pa_full, "platt_b": pb_full, "alpha": best_alpha, "threshold": th_full},
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
    c_len = len(char_tfidf.vocabulary_)
    w_len = len(word_tfidf.vocabulary_)
    lf = lr_lite_a.coef_[0][c_len + w_len :]
    ff = lr_full_a.coef_[0][384 + c_len + w_len :]
    print("\nLearned forensic weights (Lite vs Full):")
    for idx, fname in enumerate(FEATURE_NAMES):
        print(f"  {idx:2d} {fname:30s} lite={lf[idx]:+7.3f}  full={ff[idx]:+7.3f}")
    print("Training complete!")


if __name__ == "__main__":
    main()
