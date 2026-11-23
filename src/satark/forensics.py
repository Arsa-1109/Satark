"""Layer 1: Exact 35-Feature Indian Forensics Vector (f[0..34]) & 4-Pillar X-Ray (SDD §2.2).

CRITICAL INVARIANT: FEATURE_NAMES[0..34] order is frozen across Python and JS.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from .normalizer import NormalizedResult

FEATURE_NAMES: tuple[str, ...] = (
    "has_url",                      # 0
    "url_count",                    # 1
    "has_shortener",                # 2
    "has_suspicious_tld",           # 3
    "has_raw_ip_url",               # 4
    "has_punycode_url",             # 5
    "has_official_domain",          # 6
    "brand_domain_mismatch",        # 7
    "has_apk_lure",                 # 8
    "has_remote_access_app",        # 9
    "has_upi_uri",                  # 10
    "upi_collect_cashback_trap",    # 11
    "has_whatsapp_telegram_link",   # 12
    "has_indian_phone_callback",    # 13
    "has_tollfree_official",        # 14
    "has_currency_amount",          # 15
    "has_large_amount",             # 16
    "urgency_score",                # 17
    "threat_authority_score",       # 18
    "reward_lure_score",            # 19
    "kyc_bank_lex_score",           # 20
    "loan_credit_lex_score",        # 21
    "job_invest_lex_score",         # 22
    "asks_otp_or_pin",              # 23
    "has_otp_do_not_share",         # 24
    "has_dlt_header",               # 25
    "has_dlt_transactional_suffix", # 26
    "has_bank_tx_template",         # 27
    "has_delivery_template",        # 28
    "has_casual_hinglish_chat",     # 29
    "obfuscation_count",            # 30
    "upper_ratio",                  # 31
    "digit_ratio",                  # 32
    "excl_ratio",                   # 33
    "log_char_len",                 # 34
)

SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "t.ly", "cutt.ly", "rb.gy",
    "is.gd", "ow.ly", "shorturl.at", "goo.gl", "rebrand.ly", "s.id",
})

SUSPICIOUS_TLDS = (
    ".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf", ".gq",
    ".buzz", ".vip", ".club", ".shop", ".online", ".site", ".live",
    ".icu", ".loan", ".work",
)

OFFICIAL_DOMAINS = frozenset({
    "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com", "kotak.com",
    "amazon.in", "flipkart.com", "irctc.co.in", "swiggy.com", "zomato.com",
    "bluedart.com", "delhivery.com", "npci.org.in", "rbi.org.in",
    "onlinesbi.sbi", "pnbindia.in", "bankofbaroda.in", "uidai.gov.in",
    "epfindia.gov.in", "indiapost.gov.in", "cybercrime.gov.in",
    "tneu.in", "titan.co.in", "tanishq.co.in", "croma.com", "tatacliq.com",
    "myntra.com", "nykaa.com", "lenskart.com", "bookmyshow.com", "jiomart.com",
    "bigbasket.com", "blinkit.com", "zepto.in", "airtel.in", "jio.com", "vi.in",
    "makemytrip.com", "goibibo.com", "uber.com", "olacabs.com", "urbancompany.com",
    "dominos.co.in", "decathlon.in", "bsesdelhi.com", "bescom.co.in", "tatapower.com",
    "bseindia.com", "nseindia.com", "sebi.gov.in", "trai.gov.in", "dot.gov.in",
    "github.com", "linkedin.com", "naukri.com", "internshala.com", "zerodha.com",
    "groww.in", "cred.club", "jiocinema.com", "hotstar.com", "youtu.be",
    "youtube.com", "forms.gle", "google.com",
    "sihub.in", "billdesk.com", "hdfcbk.io", "kotak.in", "axbk.in",
    "sbi.in", "icicibank.in", "canarabank.com", "unionbankofindia.co.in",
    "federalbank.co.in", "yesbank.in", "indusind.com", "idfcfirstbank.com",
})

BRAND_KEYWORDS = (
    "sbi", "hdfc", "icici", "axis", "pnb", "bob", "kotak",
    "paytm", "phonepe", "gpay", "rbi", "npci", "epfo", "uidai",
    "aadhaar", "indiapost", "bescom", "msedcl", "trai", "cbi",
)

URGENCY_LEX = (
    "urgent", "immediately", "24 hours", "24 hrs", "blocked today", "expires today",
    "ends today", "deactivated today", "cut tonight", "expire",
    "suspended", "blocked", "deactivated", "last warning", "final notice",
    "turant", "abhi", "band", "jaldi", "aaj", "warn",
)

THREAT_AUTH_LEX = (
    "cbi", "police", "arrest", "warrant", "court", "customs", "fedex",
    "narcotics", "money laundering", "trai", "disconnection", "legal action",
    "fir", "freeze", "seizure",
)

REWARD_LEX = (
    "congratulations", "winner", "lottery", "kbc", "lucky draw", "jackpot",
    "prize", "won", "cashback", "free", "bonus", "spin", "jeeta", "inam", "gift",
)

KYC_BANK_LEX = (
    "kyc", "pan", "aadhaar", "yono", "netbanking", "fastag", "cvv", "atm",
    "debit card", "credit card", "account block", "update kyc", "khata",
)

LOAN_CREDIT_LEX = (
    "loan", "pre-approved", "cibil", "emi", "credit limit", "instant approval",
    "low interest", "processing fee", "karz", "sanction",
)

JOB_INVEST_LEX = (
    "part time", "work from home", "daily income", "earn per day",
    "youtube like", "subscribe", "telegram task", "hr", "trading", "crypto",
    "stock tips", "guaranteed return", "investment", "salary",
)

_URL_RE = re.compile(
    r"(?:https?://[^\s<>\"']+|www\.[^\s<>\"']+|\b[a-z0-9][a-z0-9\-]{1,62}\.(?:com|in|co\.in|org\.in|gov\.in|nic\.in|bank\.in|fin\.in|net|org|ly|xyz|top|click|tk|ml|ga|cf|gq|buzz|vip|club|shop|online|site|live|icu|loan|work|at|gd|id|gy)(?:/[^\s<>\"']*)?)",
    re.IGNORECASE,
)
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_APK_RE = re.compile(r"(?:\.apk\b|(?:download|install).{0,25}(?:app|apk|file))", re.IGNORECASE)
_REMOTE_APP_RE = re.compile(r"\b(?:anydesk|teamviewer|quicksupport|rustdesk|ammyy|screen\s*share)\b", re.IGNORECASE)
_UPI_URI_RE = re.compile(r"upi://pay\?[^\s<>\"']+", re.IGNORECASE)
_WA_TG_RE = re.compile(
    r"(?:wa\.me/|chat\.whatsapp\.com/|t\.me/|\b(?:join|contact|message|dm)\b.{0,20}\b(?:whatsapp|telegram)\b|\btelegram\s+(?:group|channel|task|hr|bot|link|id)\b)",
    re.IGNORECASE,
)
_PHONE_CALLBACK_RE = re.compile(
    r"(?:\b(?:call|contact|whatsapp|helpline|dial)\b.{0,35}(?<!\d)(?:\+91[\-\s]?)?[6-9]\d{9}(?!\d)|(?<!\d)(?:\+91[\-\s]?)?[6-9]\d{9}(?!\d).{0,35}\b(?:call|contact|whatsapp|helpline|dial)\b)",
    re.IGNORECASE,
)
_BANK_DISPUTE_PHONE_RE = re.compile(
    r"\b(?:not\s+you|if\s+not\s+done\s+by\s+you|to\s+report|to\s+dispute|sms\s+block|forward\s+this\s+sms\s+to)\b",
    re.IGNORECASE,
)
_SENDER_DLT_RE = re.compile(r"^[A-Z]{2}-[A-Z0-9]{5,6}(?:-([TSGP]))?$", re.IGNORECASE)
_SENDER_MOBILE_RE = re.compile(r"^(?:\+91[\-\s]?)?[6-9]\d{9}$")
_TOLLFREE_RE = re.compile(r"(?:1800[\-\s]?\d{3}[\-\s]?\d{4}|\b1930\b)")
_CURRENCY_RE = re.compile(r"(?:(?:rs\.?|inr|₹)\s*(\d[\d,]*(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s*(lakh|crore|k)\b)", re.IGNORECASE)
_ASKS_OTP_RE = re.compile(r"\b(?:share|send|enter|tell|provide|verify|batao|bhejo)\b.{0,20}\b(?:otp|pin|cvv|mpin|password)\b", re.IGNORECASE)
_OTP_DO_NOT_SHARE_RE = re.compile(r"\b(?:do\s*not|don'?t|never|mat|pls\s*do\s*not)\b.{0,20}\b(?:share|disclose|tell|batana)\b(?:\s.{0,20}\b(?:otp|pin|cvv|code|password)\b)?", re.IGNORECASE)
_AWARENESS_PSA_RE = re.compile(
    r"\b(?:beware\s+(?:of|dealing)|unsolicited\s+tips|take\s+informed\s+decision|use\s+telecom\s+services\s+responsibly|official\s+and\s+legitimate|rbi\s+kehta\s+hai|report\s+cyber\s*fraud|cyber\s*dost|stay\s+safe\s+from\s+fraud)\b",
    re.IGNORECASE,
)
_PHYSICAL_HANDOVER_RE = re.compile(
    r"\b(?:driver|trip|cab|ride|uber|ola|rapido|delivery\s*(?:boy|partner|agent|executive)|doorstep|at\s+the\s+gate|amazon\s+delivery|flipkart\s+delivery)\b",
    re.IGNORECASE,
)
_UPI_VPA_RE = re.compile(
    r"\b[a-z0-9.\-_]{3,}@(?:ybl|okicici|okaxis|oksbi|okhdfcbank|paytm|ibl|axl|upi|apl)\b",
    re.IGNORECASE,
)
_DLT_HEADER_RE = re.compile(r"\b[A-Z]{2}-[A-Z0-9]{5,6}(?:-[TSGP])?\b")
_DLT_TX_SUFFIX_RE = re.compile(r"\b[A-Z]{2}-[A-Z0-9]{5,6}-[TSG]\b")
_BANK_TX_RE = re.compile(
    r"(?:"
    r"\b(?:debited|credited|spent|withdrawn|auto-debited|autodebited)\b.{0,45}\b(?:a/c|ac|acct|account|card)\b.{0,25}(?:x{1,8}\d{3,4}|\*{1,4}\d{3,4}|ending\s+\d{4}|\.\.\.\d{4})"
    r"|\b(?:a/c|ac|acct|account|card)\b.{0,25}(?:x{1,8}\d{3,4}|\*{1,4}\d{3,4}|ending\s+\d{4}|\.\.\.\d{4}).{0,45}\b(?:debited|credited|spent|withdrawn|declined)\b"
    r"|\b(?:amt\s+)?sent\s+(?:rs\.?|inr|₹)\s*[\d,]+.{0,35}\b(?:from|a/c|ac)\b"
    r"|\bthank\s+you\s+for\s+using\s+your\b.{0,35}\b(?:debit|credit)\s+card\b"
    r"|\btransaction\s+of\s+(?:rs\.?|inr|₹)\s*[\d,]+.{0,50}\bcard\b.{0,30}\b(?:declined|approved)\b"
    r"|\b(?:neft|imps|rtgs)\s+transfer\s+of\s+(?:rs\.?|inr|₹)\s*[\d,]+.{0,60}\b(?:processed|credited|completed)\b"
    r"|\bcheque\s+no\.?\s*\d+.{0,50}\bcleared\b"
    r")",
    re.IGNORECASE,
)
_BANK_TX_META_RE = re.compile(r"\b(?:avail|avl|bal|balance|upi|imps|neft|rtgs|ref|refno|limit|mandate|standing|cheque|ending|declined|sent|towards|salary|rs|inr)\b", re.IGNORECASE)
_DELIVERY_ACTION_RE = re.compile(r"\b(?:out for delivery|dispatched|delivered|awb|tracking|order\s*#)\b", re.IGNORECASE)
_DELIVERY_BRAND_RE = re.compile(r"\b(?:amazon|flipkart|swiggy|zomato|bluedart|delhivery|dtdc|ekart|myntra)\b", re.IGNORECASE)
_CASUAL_HINGLISH_RE = re.compile(r"\b(?:bhai|yaar|bro|kal|mummy|papa|ghar|milte|dedunga|deta hu|bhej de|ok|haha)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ForensicsResult:
    features: list[float]  # exact length 35
    pillars: dict[str, float]
    urls: list[str]
    upi_payloads: list[dict[str, str]]
    dlt_header: str | None
    triggers: list[dict[str, str]]
    sender_analysis: dict[str, object]


def _extract_host(raw_url: str) -> str:
    candidate = raw_url if "://" in raw_url else f"http://{raw_url}"
    try:
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower().rstrip(".")
        return host
    except ValueError:
        return ""


def _is_official_host(host: str) -> bool:
    if not host:
        return False
    if host.endswith((".bank.in", ".fin.in", ".gov.in", ".nic.in")):
        return True
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def _lex_density(text_lower: str, lexicon: tuple[str, ...]) -> float:
    hits = 0
    for term in lexicon:
        if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
            hits += 1
    return min(hits / 4.0, 1.0)


def parse_upi_uris(text: str) -> list[dict[str, str]]:
    """Parse upi://pay?... URIs into structured payloads with risk notes."""
    results: list[dict[str, str]] = []
    text_lower = text.lower()
    has_lure = any(w in text_lower for w in ("receive", "cashback", "refund", "credit", "prize", "won", "claim"))
    for match in _UPI_URI_RE.finditer(text):
        raw = match.group(0)
        parsed = urlparse(raw)
        qs = parse_qs(parsed.query)
        pa = qs.get("pa", [""])[0]
        pn = qs.get("pn", [""])[0]
        am = qs.get("am", [""])[0]
        cu = qs.get("cu", ["INR"])[0]
        tn = qs.get("tn", [""])[0]
        if has_lure:
            risk_note = (
                f"Collect/Payment trap: paying ₹{am or '?'} to {pa or 'unknown VPA'} "
                "while claiming cashback/prize/refund (UPI PIN is never needed to receive money)."
            )
        else:
            risk_note = f"UPI payment request to {pa or 'unknown VPA'} for {cu} {am or 'unspecified'}."
        results.append({
            "raw": raw,
            "pa": pa,
            "pn": pn,
            "am": am,
            "cu": cu,
            "tn": tn,
            "risk_note": risk_note,
        })
    return results


def extract_forensics(norm: NormalizedResult, sender: str | None = None) -> ForensicsResult:
    """Extract the exact 35-feature vector f[0..34], 4-pillar X-ray, and trigger spans."""
    clean = norm.clean_text
    canon = norm.canonical_text  # already lowercased + Hinglish folded
    clean_lower = clean.lower()

    triggers: list[dict[str, str]] = []
    seen_triggers: set[tuple[str, str]] = set()

    def add_trigger(span: str, reason: str, polarity: str) -> None:
        key = (span.lower(), polarity)
        if span and key not in seen_triggers:
            seen_triggers.add(key)
            triggers.append({"span": span, "reason": reason, "polarity": polarity})

    # URL extraction
    raw_urls = [m.group(0).rstrip(".,);!") for m in _URL_RE.finditer(clean)]
    urls = list(dict.fromkeys(raw_urls))
    hosts = [_extract_host(u) for u in urls]

    has_url = 1.0 if len(urls) > 0 else 0.0
    url_count = min(len(urls), 5) / 5.0
    has_shortener = 1.0 if any(h in SHORTENERS for h in hosts) else 0.0
    has_suspicious_tld = 1.0 if any(any(h.endswith(tld) for tld in SUSPICIOUS_TLDS) for h in hosts) else 0.0
    has_raw_ip_url = 1.0 if any(_IPV4_RE.match(h) for h in hosts) else 0.0
    has_punycode_url = 1.0 if any("xn--" in h or any(ord(c) > 127 for c in h) for h in hosts) else 0.0
    has_official_domain = 1.0 if any(_is_official_host(h) for h in hosts) else 0.0

    for u, h in zip(urls, hosts):
        if h in SHORTENERS:
            add_trigger(u, "URL shortener obscures destination", "risk")
        elif _IPV4_RE.match(h):
            add_trigger(u, "Raw IPv4 URL instead of official domain", "risk")
        elif "xn--" in h or any(ord(c) > 127 for c in h):
            add_trigger(u, "Punycode/IDN lookalike domain", "risk")
        elif any(h.endswith(tld) for tld in SUSPICIOUS_TLDS):
            add_trigger(u, "High-risk suspicious TLD", "risk")
        elif _is_official_host(h):
            add_trigger(u, "Verified official Indian domain", "safe")

    has_wa_tg = 1.0 if _WA_TG_RE.search(clean) else 0.0
    if has_wa_tg:
        m = _WA_TG_RE.search(clean)
        if m:
            add_trigger(m.group(0), "Off-platform WhatsApp/Telegram lure", "risk")

    mentions_brand = any(re.search(r"\b" + b + r"\b", canon) for b in BRAND_KEYWORDS)
    brand_domain_mismatch = 1.0 if (mentions_brand and (has_url > 0 or has_wa_tg > 0) and has_official_domain == 0.0) else 0.0
    if brand_domain_mismatch and urls:
        add_trigger(urls[0], "Brand-domain mismatch (non-official link)", "risk")

    apk_m = _APK_RE.search(clean)
    negated_apk = bool(re.search(r"\b(?:avoid|never|do\s*not|don'?t)\b.{0,20}\b(?:download|install)", canon))
    has_apk_lure = 1.0 if (apk_m and not negated_apk) else 0.0
    if has_apk_lure and apk_m:
        add_trigger(apk_m.group(0), "Malicious APK / sideloading lure", "risk")

    remote_m = _REMOTE_APP_RE.search(clean)
    has_remote_access_app = 1.0 if remote_m else 0.0
    if remote_m:
        add_trigger(remote_m.group(0), "Remote screen-share app request", "risk")

    upi_payloads = parse_upi_uris(clean)
    has_upi_uri = 1.0 if len(upi_payloads) > 0 else 0.0
    if upi_payloads:
        add_trigger(upi_payloads[0]["raw"][:48], "Direct upi://pay payment request URI", "risk")

    upi_vpa_m = _UPI_VPA_RE.search(clean)
    has_upi_vpa = 1.0 if upi_vpa_m else 0.0

    has_upi_mention = has_upi_uri > 0 or bool(re.search(r"\b(?:upi\s*id|upi\s*pin)\b", canon))
    has_receive_lure = any(re.search(r"\b" + w + r"\b", canon) for w in ("receive", "cashback", "refund", "credit", "prize", "won", "claim"))
    urgent_transfer_cues = (
        "accidentally", "mistakenly", "refund", "urgently", "urgent", "hospital",
        "emergency", "upi is not working", "send rs", "send ₹", "transfer rs",
        "right now", "return in", "cashback", "prize", "won", "claim",
    )
    has_urgent_transfer_cue = any(cue in canon for cue in urgent_transfer_cues)
    bank_tx_m_pre = _BANK_TX_RE.search(clean)
    is_genuine_receipt = bool(bank_tx_m_pre and _BANK_TX_META_RE.search(clean)) or ("you received" in canon)
    is_upi_safety_psa = bool(re.search(r"\b(?:never\s+need\s+to\s+enter|never\s+ask|not\s+required\s+to\s+receive|safety\s+tip|npci)\b", canon))
    urgent_transfer_trap = bool(has_upi_vpa > 0 and has_urgent_transfer_cue and not is_genuine_receipt)

    upi_collect_cashback_trap = 1.0 if (((has_upi_mention and has_receive_lure) or urgent_transfer_trap) and not is_upi_safety_psa) else 0.0
    if urgent_transfer_trap and upi_vpa_m:
        add_trigger(upi_vpa_m.group(0), "Urgent / unverified UPI VPA transfer request", "risk")
    elif upi_collect_cashback_trap and not upi_payloads:
        add_trigger("UPI / cashback", "UPI payment/PIN paired with receive/cashback claim", "risk")

    toll_m = _TOLLFREE_RE.search(clean)
    has_tollfree_official = 1.0 if toll_m else 0.0
    if toll_m:
        add_trigger(toll_m.group(0), "Official 1800/1930 helpline reference", "safe")

    bank_tx_m = _BANK_TX_RE.search(clean)
    no_tx_traps = (
        has_shortener == 0.0
        and has_apk_lure == 0.0
        and has_suspicious_tld == 0.0
        and brand_domain_mismatch == 0.0
        and upi_collect_cashback_trap == 0.0
    )
    has_bank_tx_template = 1.0 if (bank_tx_m and _BANK_TX_META_RE.search(clean) and no_tx_traps) else 0.0
    if has_bank_tx_template and bank_tx_m:
        add_trigger(bank_tx_m.group(0)[:40], "Standard bank transaction alert template", "safe")

    phone_m = _PHONE_CALLBACK_RE.search(clean)
    is_bank_dispute_line = bool((has_bank_tx_template > 0.0 or has_tollfree_official > 0.0) and _BANK_DISPUTE_PHONE_RE.search(clean))
    has_indian_phone_callback = 1.0 if (phone_m and not is_bank_dispute_line) else 0.0
    if has_indian_phone_callback and phone_m:
        add_trigger(phone_m.group(0)[:36], "Personal 10-digit mobile callback number", "risk")

    has_currency_amount = 0.0
    has_large_amount = 0.0
    for cm in _CURRENCY_RE.finditer(clean):
        has_currency_amount = 1.0
        if cm.group(1):
            cleaned_num = cm.group(1).replace(",", "")
            if cleaned_num:
                val = float(cleaned_num)
                if val >= 25_000:
                    has_large_amount = 1.0
        elif cm.group(2) and cm.group(3):
            unit = cm.group(3).lower()
            num = float(cm.group(2))
            if unit in ("lakh", "crore") or (unit == "k" and num >= 25):
                has_large_amount = 1.0

    urgency_score = _lex_density(canon, URGENCY_LEX)
    threat_authority_score = _lex_density(canon, THREAT_AUTH_LEX)
    reward_lure_score = _lex_density(canon, REWARD_LEX)
    kyc_bank_lex_score = _lex_density(canon, KYC_BANK_LEX)
    loan_credit_lex_score = _lex_density(canon, LOAN_CREDIT_LEX)
    job_invest_lex_score = _lex_density(canon, JOB_INVEST_LEX)

    for lex_list, reason in (
        (URGENCY_LEX, "Artificial urgency / countdown pressure"),
        (THREAT_AUTH_LEX, "Authority / legal threat coercion"),
        (REWARD_LEX, "Reward / prize / cashback lure"),
    ):
        for term in lex_list:
            m = re.search(r"\b" + re.escape(term) + r"\b", clean_lower)
            if m:
                add_trigger(m.group(0), reason, "risk")
                break

    otp_safe_m = _OTP_DO_NOT_SHARE_RE.search(canon)
    physical_m = _PHYSICAL_HANDOVER_RE.search(canon)
    asks_otp_m = _ASKS_OTP_RE.search(canon)
    asks_otp_or_pin = 1.0 if (asks_otp_m and not otp_safe_m and not physical_m) else 0.0
    if asks_otp_or_pin and asks_otp_m:
        add_trigger(asks_otp_m.group(0), "Solicits OTP / PIN / CVV", "risk")

    awareness_m = _AWARENESS_PSA_RE.search(canon)
    no_malicious_traps = (
        has_shortener == 0.0
        and has_apk_lure == 0.0
        and has_suspicious_tld == 0.0
        and brand_domain_mismatch == 0.0
        and upi_collect_cashback_trap == 0.0
        and asks_otp_or_pin == 0.0
    )
    has_otp_do_not_share = 1.0 if (otp_safe_m or (awareness_m and no_malicious_traps)) else 0.0
    if otp_safe_m:
        add_trigger(otp_safe_m.group(0), "Standard anti-fraud OTP warning", "safe")
    elif awareness_m and no_malicious_traps:
        add_trigger(awareness_m.group(0), "Public cyber-safety / regulatory awareness advisory", "safe")

    if asks_otp_or_pin > 0.0:
        has_bank_tx_template = 0.0

    dlt_m = _DLT_HEADER_RE.search(clean)
    dlt_header = dlt_m.group(0) if dlt_m else None
    has_dlt_header = 1.0 if dlt_header else 0.0
    has_dlt_transactional_suffix = 1.0 if _DLT_TX_SUFFIX_RE.search(clean) else 0.0
    if dlt_header:
        add_trigger(dlt_header, "TRAI DLT registered sender header", "safe")

    sender_clean = sender.strip() if isinstance(sender, str) and sender.strip() else None
    sender_analysis: dict[str, object] = {"raw": sender_clean, "type": None, "impersonation_risk": False}
    if sender_clean:
        sm_dlt = _SENDER_DLT_RE.match(sender_clean)
        sm_mob = _SENDER_MOBILE_RE.match(sender_clean)
        if sm_dlt:
            dlt_header = sender_clean.upper()
            has_dlt_header = 1.0
            suffix = (sm_dlt.group(1) or "").upper()
            type_map = {"S": "dlt_service", "T": "dlt_transactional", "G": "dlt_govt", "P": "dlt_promo"}
            sender_analysis["type"] = type_map.get(suffix, "dlt_service")
            if suffix in ("T", "S", "G"):
                has_dlt_transactional_suffix = 1.0
                add_trigger(dlt_header, "Verified TRAI DLT sender ID", "safe")
            elif suffix == "P" and no_malicious_traps:
                add_trigger(dlt_header, "Registered TRAI DLT promotional sender ID", "safe")
        elif sm_mob:
            sender_analysis["type"] = "personal_mobile"
            if mentions_brand or kyc_bank_lex_score > 0 or threat_authority_score > 0 or reward_lure_score > 0:
                sender_analysis["impersonation_risk"] = True
                add_trigger(sender_clean, "Personal mobile sender impersonating institutional brand", "risk")
        else:
            sender_analysis["type"] = "unknown"

    deliv_m = _DELIVERY_ACTION_RE.search(canon)
    has_delivery_template = 1.0 if (
        deliv_m
        and _DELIVERY_BRAND_RE.search(canon)
        and not any(w in canon for w in ("customs", "fee", "pay", "bit.ly", ".apk"))
    ) else 0.0
    if has_delivery_template and deliv_m:
        add_trigger(deliv_m.group(0), "Genuine e-commerce/courier delivery pattern", "safe")

    casual_m = _CASUAL_HINGLISH_RE.search(canon)
    has_casual_hinglish_chat = 1.0 if (
        casual_m
        and has_url == 0.0
        and has_apk_lure == 0.0
        and threat_authority_score == 0.0
        and has_upi_vpa == 0.0
        and upi_collect_cashback_trap == 0.0
        and asks_otp_or_pin == 0.0
        and has_indian_phone_callback == 0.0
    ) else 0.0
    if has_casual_hinglish_chat and casual_m:
        add_trigger(casual_m.group(0), "Informal Hinglish conversational marker", "safe")

    obfuscation_count = min(len(norm.obfuscations), 4) / 4.0

    alpha_chars = [c for c in clean if ("a" <= c <= "z") or ("A" <= c <= "Z")]
    upper_ratio = (
        sum(1 for c in alpha_chars if "A" <= c <= "Z") / len(alpha_chars)
        if len(alpha_chars) >= 15
        else 0.0
    )
    digit_ratio = sum(1 for c in clean if "0" <= c <= "9") / max(len(clean), 1)
    excl_ratio = min(clean.count("!"), 5) / 5.0
    log_char_len = min(math.log1p(len(clean)) / 8.5, 1.0)

    features: list[float] = [
        has_url,                      # 0
        url_count,                    # 1
        has_shortener,                # 2
        has_suspicious_tld,           # 3
        has_raw_ip_url,               # 4
        has_punycode_url,             # 5
        has_official_domain,          # 6
        brand_domain_mismatch,        # 7
        has_apk_lure,                 # 8
        has_remote_access_app,        # 9
        has_upi_uri,                  # 10
        upi_collect_cashback_trap,    # 11
        has_wa_tg,                    # 12
        has_indian_phone_callback,    # 13
        has_tollfree_official,        # 14
        has_currency_amount,          # 15
        has_large_amount,             # 16
        urgency_score,                # 17
        threat_authority_score,       # 18
        reward_lure_score,            # 19
        kyc_bank_lex_score,           # 20
        loan_credit_lex_score,        # 21
        job_invest_lex_score,         # 22
        asks_otp_or_pin,              # 23
        has_otp_do_not_share,         # 24
        has_dlt_header,               # 25
        has_dlt_transactional_suffix, # 26
        has_bank_tx_template,         # 27
        has_delivery_template,        # 28
        has_casual_hinglish_chat,     # 29
        obfuscation_count,            # 30
        upper_ratio,                  # 31
        digit_ratio,                  # 32
        excl_ratio,                   # 33
        log_char_len,                 # 34
    ]

    pillars = {
        "urgency_fear": round(min(1.0, urgency_score * 0.65 + upper_ratio * 0.20 + excl_ratio * 0.15), 3),
        "authority_spoof": round(min(1.0, threat_authority_score * 0.55 + brand_domain_mismatch * 0.45), 3),
        "reward_lure": round(min(1.0, reward_lure_score * 0.65 + has_large_amount * 0.25 + upi_collect_cashback_trap * 0.20), 3),
        "trap_vector": round(
            min(
                1.0,
                has_shortener * 0.30
                + has_suspicious_tld * 0.25
                + has_apk_lure * 0.45
                + has_remote_access_app * 0.45
                + has_upi_uri * 0.30
                + asks_otp_or_pin * 0.40
                + has_punycode_url * 0.35
                + has_raw_ip_url * 0.35
                + has_wa_tg * 0.20
                + obfuscation_count * 0.25,
            ),
            3,
        ),
        "safe_counter_signals": round(
            min(
                1.0,
                has_official_domain * 0.35
                + has_dlt_transactional_suffix * 0.25
                + has_otp_do_not_share * 0.30
                + has_bank_tx_template * 0.30
                + has_delivery_template * 0.25
                + has_casual_hinglish_chat * 0.25
                + has_tollfree_official * 0.15,
            ),
            3,
        ),
    }

    return ForensicsResult(
        features=features,
        pillars=pillars,
        urls=urls,
        upi_payloads=upi_payloads,
        dlt_header=dlt_header,
        triggers=triggers[:10],
        sender_analysis=sender_analysis,
    )
