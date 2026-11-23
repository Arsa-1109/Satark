"""Edge-case suite (PRD §5) + Gate G9 packaging checks."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import app  # noqa: E402
from satark.engine import SatarkEngine  # noqa: E402

client = TestClient(app)
engine = SatarkEngine()


def test_empty_and_whitespace_rejected() -> None:
    with pytest.raises(ValueError):
        engine.analyze("   \n\t ")
    r1 = client.post("/api/analyze", json={"text": ""})
    assert r1.status_code == 422
    r2 = client.post("/api/analyze", json={"text": "   "})
    assert r2.status_code == 422


def test_excessive_length_truncation_and_rejection() -> None:
    long_ok = "Update KYC at bit.ly/sbi-kyc " * 220  # ~6,380 chars
    assert 5000 < len(long_ok) <= 10000
    res = engine.analyze(long_ok)
    assert res["truncated"] is True
    assert res["verdict"] == "SCAM"

    too_long = "a" * 10001
    r = client.post("/api/analyze", json={"text": too_long})
    assert r.status_code == 422


def test_emojis_only_borderline_safe() -> None:
    res = engine.analyze("🎉🔥🚨💰")
    assert res["verdict"] == "SAFE"
    assert res["certainty_tier"] == "Borderline — Verify Manually"
    assert 50.0 <= res["confidence_pct"] <= 60.0


def test_bare_url_and_three_word_scam() -> None:
    r1 = engine.analyze("bit.ly/sbi-kyc")
    assert r1["verdict"] == "SCAM"
    r2 = engine.analyze("Claim ₹50k bit.ly/x")
    assert r2["verdict"] == "SCAM"


def test_devanagari_scam_native() -> None:
    msg = "प्रिय ग्राहक, आपका SBI खाता KYC न होने के कारण आज बंद हो जाएगा। तुरंत अपडेट करें: bit.ly/sbi-kyc"
    res = engine.analyze(msg)
    assert res["verdict"] == "SCAM"


def test_official_dlt_safe_vs_spoofed_dlt_scam() -> None:
    genuine = "VM-HDFCBK-S: Rs 1,450.00 debited from A/c XX4821 on 23-11-26 via UPI/Ref 928374651. Avail Bal: Rs 24,590. Not you? Call 18002586161 or visit hdfcbank.com"
    spoofed = "VM-SBIKYC-S: Your SBI YONO account is blocked! Download update.apk at bit.ly/sbi-kyc immediately."
    assert engine.analyze(genuine)["verdict"] == "SAFE"
    assert engine.analyze(spoofed)["verdict"] == "SCAM"


def test_benign_retail_promo_sms_safe() -> None:
    titan = "Hi Arya, add something new to your Diwali look. See the latest watches at Titan World,   https://m.tneu.in/TITNWO/JeOf3mn Visit today."
    fabindia = "Hi Rohan, refresh your festive wardrobe at FabIndia. Explore new kurtas at https://m.fabindia.in/DIWALI/k8p2m Visit our store today."
    festive_scam = "Diwali Bumper Offer! Claim FREE Rs 50,000 cashback! Enter UPI PIN at bit.ly/diwali-claim immediately."
    assert engine.analyze(titan)["verdict"] == "SAFE"
    assert engine.analyze(fabindia)["verdict"] == "SAFE"
    assert engine.analyze(festive_scam)["verdict"] == "SCAM"


def test_gate_g9_packaging_and_zero_cdn() -> None:
    summary_path = ROOT / "models" / "lite" / "benchmarks_summary.json"
    assert summary_path.exists(), "models/lite/benchmarks_summary.json missing"

    dist_dir = ROOT / "frontend" / "dist"
    assert dist_dir.exists(), "frontend/dist missing — run npm run build"

    # Zero CDN URLs in frontend/dist (ignore bare protocol prefixes & standard w3.org/react.dev docs)
    cdn_re = re.compile(r"https?://(?!127\.0\.0\.1|localhost)[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)
    allowed_prefixes = ("http://www.w3.org", "https://www.w3.org", "https://react.dev", "https://reactjs.org", "https://tailwindcss.com")
    for f in dist_dir.rglob("*"):
        if f.is_file() and f.suffix in {".html", ".js", ".css", ".json", ".webmanifest"}:
            if f.name == "model_bundle.js":
                continue
            content = f.read_text(encoding="utf-8", errors="ignore")
            matches = [m.group(0) for m in cdn_re.finditer(content)]
            non_allowed = [m for m in matches if not m.lower().startswith(allowed_prefixes)]
            assert not non_allowed, f"External URL found in {f}: {non_allowed}"

    # Zero tracked/unignored files > 100 MB
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    for rel in proc.stdout.splitlines():
        p = ROOT / rel
        if p.is_file():
            assert p.stat().st_size <= 100 * 1024 * 1024, f"File exceeds 100 MB: {rel}"
