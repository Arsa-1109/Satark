"""Layer 1: Adversarial & Indic Phonetic Normaliser (SDD §2.1).

Deterministic pipeline mirrored 1-to-1 in frontend/src/engine/localEngine.js.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_RAW_CHARS = 10_000
TRUNCATE_CHARS = 5_000

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

_CONFUSABLES: dict[str, str] = {
    # Cyrillic lowercase & uppercase
    "\u0430": "a", "\u0410": "A",
    "\u0435": "e", "\u0415": "E",
    "\u043e": "o", "\u041e": "O",
    "\u0440": "p", "\u0420": "P",
    "\u0441": "c", "\u0421": "C",
    "\u0445": "x", "\u0425": "X",
    "\u0443": "y", "\u0423": "Y",
    "\u0456": "i", "\u0406": "I",
    "\u043a": "k", "\u041a": "K",
    "\u043c": "m", "\u041c": "M",
    "\u0442": "t", "\u0422": "T",
    "\u043d": "h", "\u041d": "H",
    "\u0432": "b", "\u0412": "B",
    # Greek lowercase & uppercase
    "\u03b1": "a", "\u0391": "A",
    "\u03b5": "e", "\u0395": "E",
    "\u03bf": "o", "\u039f": "O",
    "\u03c1": "p", "\u03a1": "P",
    "\u03ba": "k", "\u039a": "K",
    "\u03c4": "t", "\u03a4": "T",
    "\u03b9": "i", "\u0399": "I",
}

_SPACED_TOKEN_RE = re.compile(r"\b([a-zA-Z])(?:[\.\-_\s]{1,2}([a-zA-Z])){2,7}\b")
_SPACED_PART_RE = re.compile(r"\b([a-zA-Z](?:[\.\-_\s]{1,2}[a-zA-Z]){2,7})\b")

_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
}

_PURE_CODE_RE = re.compile(
    r"^(?:xx+\d+|\*+\d+|rs\.?\d+|\d{1,4}[a-z]{1,4}|[a-z]{2,}\d{1,5}|[a-z]{1,3}\d{3,}|v\d+(?:\.\d+)*)$",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9@$]+")

_HINGLISH_MAP: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:paise|pese|paisaa)\b", re.IGNORECASE), "paisa"),
    (re.compile(r"\b(?:khaata|khta)\b", re.IGNORECASE), "khata"),
    (re.compile(r"\b(?:turnt|turanth)\b", re.IGNORECASE), "turant"),
    (re.compile(r"\b(?:jita|jeete|jeeto)\b", re.IGNORECASE), "jeeta"),
    (re.compile(r"\b(?:bhejein|bhejdo|bhejde)\b", re.IGNORECASE), "bhejo"),
    (re.compile(r"\b(?:bnd|bandh)\b", re.IGNORECASE), "band"),
    (re.compile(r"\b(?:apka|aapke|apke)\b", re.IGNORECASE), "aapka"),
    (re.compile(r"\b(?:khatam|khtm)\b", re.IGNORECASE), "khatam"),
)


@dataclass(frozen=True)
class NormalizedResult:
    raw_text: str
    clean_text: str
    canonical_text: str
    obfuscations: list[str]
    truncated: bool


def _collapse_spaced(match: re.Match[str]) -> str:
    return re.sub(r"[\.\-_\s]+", "", match.group(0))


def _decode_leet_token(token: str) -> tuple[str, bool]:
    # Must contain both letters and leet chars, and not look like a masked acct / pure code / Rs500
    if _PURE_CODE_RE.match(token):
        return token, False
    has_alpha = any("a" <= c.lower() <= "z" for c in token)
    has_leet = any(c in _LEET_MAP for c in token)
    if not (has_alpha and has_leet):
        return token, False
    decoded = "".join(_LEET_MAP.get(c, c) for c in token)
    return decoded, (decoded != token)


def normalize_text(text: str, *, validate: bool = True) -> NormalizedResult:
    """Apply the 7-step SDD §2.1 normalisation sequence."""
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if validate:
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")
        if len(text) > MAX_RAW_CHARS:
            raise ValueError(f"Input text exceeds maximum length of {MAX_RAW_CHARS} characters.")

    raw_text = text
    truncated = False
    working = raw_text
    if len(working) > TRUNCATE_CHARS:
        working = working[:TRUNCATE_CHARS]
        truncated = True

    obfuscations: list[str] = []

    # Step 2: Strip zero-width / bidi characters
    stripped = _ZERO_WIDTH_RE.sub("", working)
    if stripped != working:
        obfuscations.append("zero_width")
    working = stripped

    # Step 3: Unicode NFKC
    working = unicodedata.normalize("NFKC", working)

    # Step 4: Cyrillic/Greek confusable folding
    folded_chars: list[str] = []
    had_homoglyph = False
    for ch in working:
        mapped = _CONFUSABLES.get(ch)
        if mapped is not None:
            folded_chars.append(mapped)
            had_homoglyph = True
        else:
            folded_chars.append(ch)
    if had_homoglyph:
        obfuscations.append("cyrillic_homoglyph")
    working = "".join(folded_chars)

    # Step 5: Collapse spaced/dotted single-letter tokens (K.Y.C -> kyc, O T P -> otp)
    collapsed = _SPACED_PART_RE.sub(_collapse_spaced, working)
    if collapsed != working:
        obfuscations.append("spaced_token")
    working = collapsed

    # Step 6: Token-level leetspeak decode (preserve URLs/emails by skipping tokens inside URLs)
    had_leet = False
    url_spans = [
        (m.start(), m.end())
        for m in re.finditer(
            r"(?:https?://|upi://|www\.)\S+|\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.[a-z]{2,}/\S+",
            working,
            re.IGNORECASE,
        )
    ]

    def _in_url(idx: int) -> bool:
        return any(s <= idx < e for s, e in url_spans)

    def _leet_sub(m: re.Match[str]) -> str:
        nonlocal had_leet
        if _in_url(m.start()):
            return m.group(0)
        dec, changed = _decode_leet_token(m.group(0))
        if changed:
            had_leet = True
        return dec

    working = _TOKEN_RE.sub(_leet_sub, working)
    if had_leet:
        obfuscations.append("leetspeak")

    # Tag excessive caps (>=15 alpha chars and >65% uppercase)
    alpha_chars = [c for c in working if ("a" <= c <= "z") or ("A" <= c <= "Z")]
    if len(alpha_chars) >= 15:
        upper_cnt = sum(1 for c in alpha_chars if "A" <= c <= "Z")
        if upper_cnt / len(alpha_chars) > 0.65:
            obfuscations.append("excessive_caps")

    clean_text = re.sub(r"[ \t]+", " ", working).strip()

    # Step 7: Indic Hinglish phonetic canonicalisation
    canonical = clean_text.lower()
    for pat, repl in _HINGLISH_MAP:
        canonical = pat.sub(repl, canonical)

    return NormalizedResult(
        raw_text=raw_text,
        clean_text=clean_text,
        canonical_text=canonical,
        obfuscations=obfuscations,
        truncated=truncated,
    )
