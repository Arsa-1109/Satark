"""CLI for Satark — 100% offline on-device Indian scam message detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from satark.clipboard import _read_windows_clipboard, is_scannable_clipboard_text  # noqa: E402
from satark.engine import SatarkEngine  # noqa: E402


def _format_verdict(res: dict, use_color: bool) -> str:
    verdict = res["verdict"]
    is_borderline = res.get("certainty_tier", "").startswith("Borderline")
    if use_color:
        if is_borderline:
            verdict_str = f"\033[93m{verdict} (BORDERLINE)\033[0m"
        elif verdict == "SCAM":
            verdict_str = f"\033[91m{verdict}\033[0m"
        else:
            verdict_str = f"\033[92m{verdict}\033[0m"
    else:
        verdict_str = verdict

    lines: list[str] = []
    cat_str = f" [{res['category']}" + (f" / {res['sub_type']}]" if res["sub_type"] else "]") if res["category"] else ""
    lines.append(f"Verdict    : {verdict_str}{cat_str}")
    lines.append(f"Confidence : {res['confidence_pct']}% ({res['certainty_tier']}, p_scam={res['scam_probability']})")
    lines.append(f"Mode/Speed : {res['model_mode']} in {res['latency_ms']} ms")
    if res["obfuscations_detected"]:
        lines.append(f"Obfuscation: {', '.join(res['obfuscations_detected'])}")
    if res["forensics"]["triggers"]:
        lines.append("Signals    :")
        for t in res["forensics"]["triggers"]:
            lines.append(f"  - [{t['polarity'].upper()}] {t['span']} -> {t['reason']}")
    return "\n".join(lines)


def _print_result(res: dict, args: argparse.Namespace, use_color: bool) -> None:
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if args.report:
        draft = res.get("report_draft_1930")
        if draft:
            print(draft)
        else:
            print("No 1930 / Chakshu report draft generated (message classified as SAFE).")
        return
    print(_format_verdict(res, use_color))


def _run_interactive(engine: SatarkEngine, args: argparse.Namespace, use_color: bool) -> None:
    print("Satark Interactive Mode (100% Offline). Type a message and press Enter; type 'quit' or 'exit' to stop.")
    while True:
        try:
            line = input("\nsatark> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        msg = line.strip()
        if not msg:
            continue
        if msg.lower() in {"quit", "exit"}:
            break
        res = engine.analyze(msg, source="text", force_lite=args.lite, sender=args.sender)
        _print_result(res, args, use_color)


def _run_watch_clipboard(engine: SatarkEngine, args: argparse.Namespace, use_color: bool) -> None:
    if sys.platform != "win32":
        print("Clipboard watcher is only supported on Windows.", file=sys.stderr)
        sys.exit(1)

    print("Satark Clipboard Watcher active (Ctrl+C to stop)...")
    last_hash: str | None = None
    initial = _read_windows_clipboard()
    if initial:
        last_hash = hashlib.sha256(initial.encode("utf-8", errors="ignore")).hexdigest()

    try:
        while True:
            text = _read_windows_clipboard()
            if text and text.strip():
                digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                if digest != last_hash:
                    last_hash = digest
                    if is_scannable_clipboard_text(text):
                        print("\n--- Clipboard Scan ---")
                        res = engine.analyze(text, source="clipboard", force_lite=args.lite, sender=args.sender)
                        _print_result(res, args, use_color)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped clipboard watcher.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Satark — On-Device Indian Scam Detector")
    parser.add_argument("text", nargs="?", help="Message text to analyze")
    parser.add_argument("--file", "-f", type=Path, help="Read message text from a file")
    parser.add_argument("--sender", "-s", default=None, help="Optional SMS Sender ID (e.g. VM-HDFCBK-S or +919876543210)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON contract")
    parser.add_argument("--lite", action="store_true", help="Force Lite Fallback mode")
    parser.add_argument("--report", action="store_true", help="Print PII-redacted 1930 / Chakshu report draft")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive REPL loop")
    parser.add_argument("--watch-clipboard", action="store_true", help="Watch Windows clipboard and scan new messages")
    args = parser.parse_args()

    use_color = sys.stdout.isatty() and not args.json
    engine = SatarkEngine(force_lite=args.lite)

    if args.interactive:
        _run_interactive(engine, args, use_color)
        return

    if args.watch_clipboard:
        _run_watch_clipboard(engine, args, use_color)
        return

    if args.file:
        msg = args.file.read_text(encoding="utf-8")
    elif args.text is not None:
        msg = args.text
    elif not sys.stdin.isatty():
        msg = sys.stdin.read()
    else:
        parser.error("Provide message text as argument, --file, --interactive, --watch-clipboard, or stdin.")
        return

    res = engine.analyze(msg, source="text", force_lite=args.lite, sender=args.sender)
    _print_result(res, args, use_color)


if __name__ == "__main__":
    main()
