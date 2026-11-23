"""Layer 2: Dual-Mode Calibrated Inference & Entropy-Gated Other Routing (SDD §2.4)."""

from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from .backbone import BackboneNotAvailableError, OnnxBackbone
from .forensics import ForensicsResult, extract_forensics
from .normalizer import NormalizedResult, normalize_text
from .redact import build_1930_draft

CATEGORIES: tuple[str, ...] = (
    "KYC/Bank",
    "Lottery/Prize",
    "Loan/Credit",
    "Job/Investment",
    "Other",
)

_ROOT = Path(__file__).resolve().parent.parent.parent


def _platt_prob(z: float, a: float, b: float) -> float:
    val = a * z + b
    if val >= 0:
        ez = math.exp(-val)
        return 1.0 / (1.0 + ez)
    ez = math.exp(val)
    return ez / (1.0 + ez)


def _forensic_risk_score(f: list[float]) -> float:
    """Independent forensic risk score p_for in [0, 1] for Dual-View Disagreement."""
    risk = (
        f[2] * 0.25   # has_shortener
        + f[3] * 0.20 # has_suspicious_tld
        + f[4] * 0.30 # has_raw_ip_url
        + f[5] * 0.30 # has_punycode_url
        + f[7] * 0.35 # brand_domain_mismatch
        + f[8] * 0.45 # has_apk_lure
        + f[9] * 0.40 # has_remote_access_app
        + f[10] * 0.25 # has_upi_uri
        + f[11] * 0.35 # upi_collect_cashback_trap
        + f[12] * 0.15 # has_whatsapp_telegram_link
        + f[17] * 0.25 # urgency_score
        + f[18] * 0.25 # threat_authority_score
        + f[19] * 0.25 # reward_lure_score
        + f[23] * 0.40 # asks_otp_or_pin
        + f[30] * 0.20 # obfuscation_count
    )
    safe = (
        f[6] * 0.30   # has_official_domain
        + f[14] * 0.15 # has_tollfree_official
        + f[24] * 0.35 # has_otp_do_not_share
        + f[26] * 0.15 # has_dlt_transactional_suffix
        + f[27] * 0.30 # has_bank_tx_template
        + f[28] * 0.25 # has_delivery_template
        + f[29] * 0.25 # has_casual_hinglish_chat
    )
    raw = 0.5 + (risk - safe)
    return max(0.0, min(1.0, raw))


def _infer_sub_type(clean_text: str, forensics: ForensicsResult, category: str) -> str | None:
    low = clean_text.lower()
    f = forensics.features
    if f[8] > 0 or ".apk" in low:
        return "Malicious APK"
    if any(w in low for w in ("digital arrest", "cbi", "narcotics", "money laundering", "warrant", "cyber cell")):
        return "Digital Arrest"
    if f[11] > 0 or f[10] > 0:
        return "UPI Collect Trap"
    if any(w in low for w in ("electricity", "bijli", "bescom", "msedcl", "disconnect", "trai", "customs", "parcel")):
        return "Utility Threat"
    if f[9] > 0 or f[7] > 0:
        return "Impersonation"
    return None


class SatarkEngine:
    """Dual-mode calibrated Satark scam detector."""

    def __init__(
        self,
        lite_dir: Path | None = None,
        onnx_dir: Path | None = None,
        force_lite: bool = False,
    ) -> None:
        self.lite_dir = Path(lite_dir) if lite_dir else _ROOT / "models" / "lite"
        self.onnx_dir = Path(onnx_dir) if onnx_dir else _ROOT / "models" / "onnx"

        vec_bundle = joblib.load(self.lite_dir / "vectorizers.joblib")
        self.char_tfidf = vec_bundle["char_tfidf"]
        self.word_tfidf = vec_bundle["word_tfidf"]
        self.forensic_scaler = vec_bundle["forensic_scaler"]

        self.stage_a_lite = joblib.load(self.lite_dir / "stage_a_lite.joblib")
        self.stage_b_lite = joblib.load(self.lite_dir / "stage_b_lite.joblib")

        self.backbone: OnnxBackbone | None = None
        self.stage_a_full: dict[str, Any] | None = None
        self.stage_b_full: dict[str, Any] | None = None
        self.mode = "lite_fallback"
        self.model_sha256 = self._sha256_file(self.lite_dir / "stage_a_lite.joblib")

        if not force_lite:
            full_a = self.onnx_dir / "stage_a_full.joblib"
            full_b = self.onnx_dir / "stage_b_full.joblib"
            if full_a.exists() and full_b.exists():
                try:
                    self.backbone = OnnxBackbone(self.onnx_dir, max_length=128)
                    self.stage_a_full = joblib.load(full_a)
                    self.stage_b_full = joblib.load(full_b)
                    self.mode = "full_onnx_hybrid"
                    self.model_sha256 = self.backbone.get_sha256()
                except BackboneNotAvailableError:
                    self.backbone = None
                    self.mode = "lite_fallback"

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _build_sparse_features(self, canonical_texts: list[str], forensic_matrix: np.ndarray) -> csr_matrix:
        xc = self.char_tfidf.transform(canonical_texts)
        xw = self.word_tfidf.transform(canonical_texts)
        xf = csr_matrix(self.forensic_scaler.transform(forensic_matrix).astype(np.float32))
        return hstack([xc, xw, xf], format="csr")

    def _predict_raw(
        self,
        clean_texts: list[str],
        canonical_texts: list[str],
        forensic_matrix: np.ndarray,
        use_full: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_sparse = self._build_sparse_features(canonical_texts, forensic_matrix)

        if use_full and self.backbone is not None and self.stage_a_full is not None and self.stage_b_full is not None:
            embs = self.backbone.encode(clean_texts)
            x_hybrid = hstack([csr_matrix(embs), x_sparse], format="csr")
            lr_a = self.stage_a_full["lr"]
            lgbm_a = self.stage_a_full["lgbm"]
            platt_a = self.stage_a_full["platt_a"]
            platt_b = self.stage_a_full["platt_b"]
            alpha = float(self.stage_a_full.get("alpha", 0.5))

            lr_logits = lr_a.decision_function(x_hybrid)
            xf_scaled = self.forensic_scaler.transform(forensic_matrix).astype(np.float32)
            lgbm_in = np.hstack([embs, xf_scaled, lr_logits[:, None]])
            lgbm_probs = np.clip(lgbm_a.predict_proba(lgbm_in)[:, 1], 1e-6, 1.0 - 1e-6)
            lgbm_logits = np.log(lgbm_probs / (1.0 - lgbm_probs))

            blended_logits = alpha * lr_logits + (1.0 - alpha) * lgbm_logits
            p_scam = np.array([_platt_prob(float(z), platt_a, platt_b) for z in blended_logits], dtype=np.float64)
            p_cats = self.stage_b_full["clf"].predict_proba(x_hybrid)
            return p_scam, p_cats

        # Lite fallback
        clf_a = self.stage_a_lite["clf"]
        platt_a = self.stage_a_lite["platt_a"]
        platt_b = self.stage_a_lite["platt_b"]
        logits = clf_a.decision_function(x_sparse)
        p_scam = np.array([_platt_prob(float(z), platt_a, platt_b) for z in logits], dtype=np.float64)
        p_cats = self.stage_b_lite["clf"].predict_proba(x_sparse)
        return p_scam, p_cats

    def analyze(self, text: str, source: str = "text", *, force_lite: bool = False, sender: str | None = None) -> dict[str, Any]:
        """Run full Layer 1 + Layer 2 inference and return PRD §3 JSON contract."""
        t0 = time.perf_counter()
        norm: NormalizedResult = normalize_text(text, validate=True)
        forensics: ForensicsResult = extract_forensics(norm, sender=sender)
        f = forensics.features

        use_full = (self.mode == "full_onnx_hybrid") and not force_lite
        active_mode = "lite_fallback" if force_lite else self.mode

        # Low-signal guard: <2 alphanumeric chars and no URL
        alnum_count = len(re.findall(r"[A-Za-z0-9\u0900-\u0d7f]", norm.clean_text))
        if alnum_count < 2 and f[0] == 0.0:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "verdict": "SAFE",
                "category": None,
                "sub_type": None,
                "confidence_pct": 55.0,
                "scam_probability": 0.45,
                "certainty_tier": "Borderline — Verify Manually",
                "model_mode": active_mode,
                "truncated": norm.truncated,
                "latency_ms": latency_ms,
                "normalised_text": norm.clean_text,
                "obfuscations_detected": norm.obfuscations,
                "pillars": forensics.pillars,
                "forensics": {
                    "urls": forensics.urls,
                    "upi_payloads": forensics.upi_payloads,
                    "dlt_header": forensics.dlt_header,
                    "sender_analysis": forensics.sender_analysis,
                    "triggers": forensics.triggers,
                },
                "report_draft_1930": None,
            }

        f_mat = np.array([f], dtype=np.float32)
        p_scam_arr, p_cats_arr = self._predict_raw(
            [norm.clean_text],
            [norm.canonical_text],
            f_mat,
            use_full=use_full,
        )
        p_scam = float(p_scam_arr[0])
        p_cats = p_cats_arr[0]

        # Forensic Override Guard (Anti-Spoof)
        if (f[8] > 0 or f[7] > 0 or f[11] > 0 or f[23] > 0) and p_scam < 0.65:
            p_scam = max(p_scam, 0.78)
        if forensics.sender_analysis.get("impersonation_risk"):
            p_scam = max(p_scam, 0.85)

        # Trusted-Evidence Guard (verified official links + zero risk triggers)
        risk_trigger_count = sum(1 for t in forensics.triggers if t["polarity"] == "risk")
        if (
            f[6] > 0
            and forensics.all_urls_official
            and risk_trigger_count == 0
            and f[7] == 0
            and f[8] == 0
            and f[9] == 0
            and f[10] == 0
            and f[11] == 0
            and f[13] == 0
            and f[23] == 0
            and not forensics.sender_analysis.get("impersonation_risk")
        ):
            p_scam = min(p_scam, 0.15)

        threshold = float((self.stage_a_full if use_full else self.stage_a_lite).get("threshold", 0.5))
        verdict = "SCAM" if p_scam >= threshold else "SAFE"
        confidence_pct = round(max(p_scam, 1.0 - p_scam) * 100.0, 1)

        # Dual-View Borderline Band
        p_for = _forensic_risk_score(f)
        ml_uncertain = 0.38 <= p_scam <= 0.68
        forensic_flag_on_safe = p_scam < 0.38 and p_for >= 0.65
        safe_counter_on_scam = 0.50 <= p_scam < 0.85 and p_for <= 0.25
        if ml_uncertain or forensic_flag_on_safe or safe_counter_on_scam:
            certainty_tier = "Borderline — Verify Manually"
            if not ml_uncertain:
                disagreement_cap = round(max(50.0, 100.0 - abs(p_scam - p_for) * 65.0), 1)
                confidence_pct = min(confidence_pct, disagreement_cap)
        elif confidence_pct < 82.0:
            certainty_tier = "Moderate"
        else:
            certainty_tier = "High Certainty"

        category: str | None = None
        sub_type: str | None = None
        report_draft: str | None = None

        if verdict == "SCAM":
            classes = (self.stage_b_full if use_full else self.stage_b_lite)["classes"]
            max_idx = int(np.argmax(p_cats))
            max_p = float(p_cats[max_idx])
            eps = 1e-12
            entropy = -float(np.sum(p_cats * np.log(np.clip(p_cats, eps, 1.0))))
            h_norm = entropy / math.log(len(classes))
            if max_p < 0.45 or h_norm > 0.82:
                category = "Other"
            else:
                category = str(classes[max_idx])
            sub_type = _infer_sub_type(norm.clean_text, forensics, category)
            report_draft = build_1930_draft(
                norm.raw_text,
                category,
                sub_type,
                forensics.urls,
                forensics.upi_payloads,
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "verdict": verdict,
            "category": category,
            "sub_type": sub_type,
            "confidence_pct": confidence_pct,
            "scam_probability": round(p_scam, 4),
            "certainty_tier": certainty_tier,
            "model_mode": active_mode,
            "truncated": norm.truncated,
            "latency_ms": latency_ms,
            "normalised_text": norm.clean_text,
            "obfuscations_detected": norm.obfuscations,
            "pillars": forensics.pillars,
            "forensics": {
                "urls": forensics.urls,
                "upi_payloads": forensics.upi_payloads,
                "dlt_header": forensics.dlt_header,
                "sender_analysis": forensics.sender_analysis,
                "triggers": forensics.triggers,
            },
            "report_draft_1930": report_draft,
        }
