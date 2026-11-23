"""PII-Redacted 1930 / Sanchar Saathi Chakshu Incident Draft Generator (SDD §2.5)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_AADHAAR_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_OTP_RE = re.compile(r"(?i)(\b(?:otp|pin|code|cvv)\b[\s:is\-]*)(\d{4,6})\b")
_ACCT_RE = re.compile(r"(?i)(\b(?:a/c|acct|account|card)\b[\s:#no.]*[x*]*?)(\d{4,16})\b")


def redact_victim_pii(text: str) -> str:
    """Mask victim Aadhaar, PAN, OTPs, and account numbers while keeping scammer indicators."""
    out = _AADHAAR_RE.sub("[AADHAAR_REDACTED]", text)
    out = _PAN_RE.sub("[PAN_REDACTED]", out)
    out = _OTP_RE.sub(r"\1[OTP_REDACTED]", out)
    out = _ACCT_RE.sub(r"\1[ACCT_REDACTED]", out)
    return out


def build_1930_draft(
    raw_text: str,
    category: str | None,
    sub_type: str | None,
    urls: list[str],
    upi_payloads: list[dict[str, str]],
) -> str:
    """Format a copyable PII-redacted incident report for Cybercrime 1930 / Chakshu."""
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M IST")
    redacted_msg = redact_victim_pii(raw_text.strip())
    cat_label = category or "Suspected Scam"
    sub_label = f" ({sub_type})" if sub_type else ""
    url_line = ", ".join(urls) if urls else "None detected"
    vpas = [p["pa"] for p in upi_payloads if p.get("pa")]
    upi_line = ", ".join(vpas) if vpas else "None detected"

    return (
        f"Sanchar Saathi Chakshu / 1930 Cybercrime Report Draft\n"
        f"Timestamp: {ts}\n"
        f"Detected Category: {cat_label}{sub_label}\n"
        f"Suspect URLs: {url_line}\n"
        f"Suspect UPI VPAs: {upi_line}\n"
        f"Redacted Message Excerpt:\n"
        f"\"\"\"\n{redacted_msg[:600]}\n\"\"\""
    )
