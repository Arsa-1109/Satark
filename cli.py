"""CLI for Satark — 100% offline on-device Indian scam message detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from satark.engine import SatarkEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Satark — On-Device Indian Scam Detector")
    parser.add_argument("text", nargs="?", help="Message text to analyze")
    parser.add_argument("--file", "-f", type=Path, help="Read message text from a file")
    parser.add_argument("--sender", "-s", default=None, help="Optional SMS Sender ID (e.g. VM-HDFCBK-S or +919876543210)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON contract")
    parser.add_argument("--lite", action="store_true", help="Force Lite Fallback mode")
    args = parser.parse_args()

    if args.file:
        msg = args.file.read_text(encoding="utf-8")
    elif args.text is not None:
        msg = args.text
    elif not sys.stdin.isatty():
        msg = sys.stdin.read()
    else:
        parser.error("Provide message text as argument, --file, or stdin.")
        return

    engine = SatarkEngine(force_lite=args.lite)
    res = engine.analyze(msg, source="text", force_lite=args.lite, sender=args.sender)

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    cat_str = f" [{res['category']}" + (f" / {res['sub_type']}]" if res["sub_type"] else "]") if res["category"] else ""
    print(f"Verdict    : {res['verdict']}{cat_str}")
    print(f"Confidence : {res['confidence_pct']}% ({res['certainty_tier']}, p_scam={res['scam_probability']})")
    print(f"Mode/Speed : {res['model_mode']} in {res['latency_ms']} ms")
    if res["obfuscations_detected"]:
        print(f"Obfuscation: {', '.join(res['obfuscations_detected'])}")
    if res["forensics"]["triggers"]:
        print("Signals    :")
        for t in res["forensics"]["triggers"]:
            print(f"  - [{t['polarity'].upper()}] {t['span']} -> {t['reason']}")


if __name__ == "__main__":
    main()
