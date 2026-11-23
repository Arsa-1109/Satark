"""Deterministic Red-Team Adversarial Mutator (SDD §2.5)."""

from __future__ import annotations

import re

_CYRILLIC_SWAPS = {
    "a": "\u0430",  # Cyrillic а
    "e": "\u0435",  # Cyrillic е
    "o": "\u043e",  # Cyrillic о
    "p": "\u0440",  # Cyrillic р
    "c": "\u0441",  # Cyrillic с
}

_LEET_WORDS = {
    "loan": "l04n",
    "bank": "b4nk",
    "free": "fr33",
    "block": "bl0ck",
    "blocked": "bl0ck3d",
    "urgent": "urg3nt",
    "verify": "v3r1fy",
    "account": "4cc0unt",
    "prize": "pr1z3",
    "cashback": "c4shb4ck",
    "update": "upd4t3",
    "salary": "s4l4ry",
}

_SPACED_WORDS = {
    "kyc": "K.Y.C",
    "otp": "O T P",
    "pan": "P.A.N",
    "sbi": "S.B.I",
    "rbi": "R.B.I",
    "upi": "U.P.I",
    "cbi": "C.B.I",
}

_HINGLISH_SWAPS = {
    "paisa": "pese",
    "paise": "pese",
    "khata": "khaata",
    "turant": "turanth",
    "jeeta": "jita",
    "bhejo": "bhejein",
    "band": "bnd",
    "aapka": "apka",
}


def mutate_message(text: str, level: str = "medium") -> str:
    """Apply deterministic adversarial obfuscations without breaking URLs."""
    # Protect URLs from mutation so trap vectors remain valid
    url_tokens: list[str] = []

    def _stash_url(m: re.Match[str]) -> str:
        url_tokens.append(m.group(0))
        return f"__URL_TOKEN_{len(url_tokens) - 1}__"

    working = re.sub(r"(?:https?://|upi://|www\.)\S+", _stash_url, text, flags=re.IGNORECASE)

    # 1. Spaced acronyms
    for src, dst in _SPACED_WORDS.items():
        working = re.sub(rf"\b{src}\b", dst, working, flags=re.IGNORECASE)

    # 2. Leetspeak on target words
    for src, dst in _LEET_WORDS.items():
        working = re.sub(rf"\b{src}\b", dst, working, flags=re.IGNORECASE)

    # 3. Hinglish drift
    for src, dst in _HINGLISH_SWAPS.items():
        working = re.sub(rf"\b{src}\b", dst, working, flags=re.IGNORECASE)

    # 4. Cyrillic homoglyphs + zero-width on medium/hard
    if level in ("medium", "hard"):
        chars: list[str] = []
        swaps = 0
        for idx, ch in enumerate(working):
            if ch in _CYRILLIC_SWAPS and swaps < 6 and idx % 5 == 0:
                chars.append(_CYRILLIC_SWAPS[ch])
                swaps += 1
            else:
                chars.append(ch)
        working = "".join(chars)

    if level == "hard":
        # Insert zero-width space inside common trigger words
        working = re.sub(r"\b(click|claim|reward|arrest)\b", lambda m: m.group(0)[:2] + "\u200b" + m.group(0)[2:], working, flags=re.IGNORECASE)

    # Restore URLs
    for idx, url in enumerate(url_tokens):
        working = working.replace(f"__URL_TOKEN_{idx}__", url)

    return working
