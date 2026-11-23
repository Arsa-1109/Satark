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


def test_realworld_service_and_subtle_scams() -> None:
    safe_samples = [
        "Dear Customer, your Jio plan of Rs 299 is expiring in 3 days. Recharge now on MyJio app or jio.com to enjoy uninterrupted services.",
        "Airtel: You have used 90% of your daily 1.5GB data quota. Buy a data pack on Airtel Thanks app.",
        "Zomato: Craving biryani tonight? Get 50% OFF up to Rs 120 using code TRYNEW. Order now on Zomato app!",
        "ICICI Bank: Your credit card bill of Rs 12,450 is due on 28 Nov 2026. Pay via iMobile app to avoid late fees.",
        "Uber: Your ride with Ramesh (KA01AB1234) is arriving in 3 mins. Share PIN 4829 with driver to start trip.",
        "Can you please share the OTP for the Amazon delivery? The delivery boy is at the gate.",
    ]
    scam_samples = [
        "Bro quick favor, my UPI is not working, can you send Rs 8000 to my friend 9811223344@paytm right now? Will return in 1 hour pakka.",
        "Sir I accidentally transferred Rs 15000 to your Google Pay instead of my brother. Please refund Rs 15000 urgently to 9876543210@ybl, my mother is in hospital.",
        "Hi, I am Priya from Amazon HR. We are hiring part-time reviewers, earn 2000-5000 daily from home. Reply YES for details.",
    ]
    for s in safe_samples:
        assert engine.analyze(s)["verdict"] == "SAFE", f"Expected SAFE for: {s}"
    for s in scam_samples:
        assert engine.analyze(s)["verdict"] == "SCAM", f"Expected SCAM for: {s}"


def test_headerless_transactional_and_sender_id() -> None:
    tx_msgs = [
        "Dear UPI user A/C X1234 debited by 450.0 on date 24Nov26 trf to ZOMATO Refno 432918472910. If not u? call 1800111109.",
        "Sent Rs.250.00 From HDFC Bank A/C *4821 To SWIGGY On 24/11/26 Ref 632819204512 Not You? Call 18002586161",
        "INR 1,299.00 spent using ICICI Bank Card XX8001 on 24-Nov-26 on AMAZON. Avl Limit: INR 84,200.00. If not you, call 18001080 or SMS BLOCK 8001 to 9215676766.",
        "HDFC Bank: Rs.199.00 auto-debited from your Card XX4421 for Spotify Standing Instruction. Manage at sihub.in",
    ]
    for m in tx_msgs:
        assert engine.analyze(m)["verdict"] == "SAFE", f"Expected SAFE for headerless tx: {m}"

    # Sender ID: DLT transactional vs personal mobile impersonation
    res_dlt = engine.analyze("Rs.1,200 debited from A/c XX4821 on 24-11-26. Avl Bal Rs 9,400.", sender="VM-HDFCBK-S")
    assert res_dlt["verdict"] == "SAFE"
    assert res_dlt["forensics"]["dlt_header"] == "VM-HDFCBK-S"

    res_spoof = engine.analyze("Dear customer, your SBI YONO account is suspended. Verify KYC immediately.", sender="+919876543210")
    assert res_spoof["verdict"] == "SCAM"
    assert res_spoof["forensics"]["sender_analysis"]["impersonation_risk"] is True


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


def test_clipboard_noise_filter_and_toggle_api() -> None:
    from satark.clipboard import is_scannable_clipboard_text

    assert is_scannable_clipboard_text("") is False
    assert is_scannable_clipboard_text("hello") is False
    assert is_scannable_clipboard_text(r"C:\Users\arsal\Desktop\file.txt") is False
    assert is_scannable_clipboard_text("import os\nfrom pathlib import Path\ndef foo(): pass") is False
    assert is_scannable_clipboard_text('{"key": "value"}') is False

    assert is_scannable_clipboard_text("bit.ly/sbi-kyc") is True
    assert is_scannable_clipboard_text("VM-SBIKYC: Your SBI YONO account is blocked! Update KYC now.") is True

    resp = client.post("/api/clipboard/toggle", json={"only_alert_scam": True, "smart_filter": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["only_alert_scam"] is True
    assert body["smart_filter"] is True

    # Reset back to default
    client.post("/api/clipboard/toggle", json={"only_alert_scam": False, "smart_filter": True})


def test_cli_report_flag() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "cli.py"),
            "--lite",
            "--report",
            "VM-SBIKYC: Aapka SBI khata band hoga! Update at bit.ly/sbi-kyc",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Sanchar Saathi Chakshu / 1930 Cybercrime Report Draft" in proc.stdout
    assert "bit.ly/sbi-kyc" in proc.stdout
