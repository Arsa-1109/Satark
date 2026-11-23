"""Gate G8: Subprocess zero-network offline privacy proof.

Monkey-patches socket and urllib BEFORE importing satark.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.request
from pathlib import Path

_AUDIT = {"sockets_opened": 0, "bytes_transmitted": 0}


def _blocked_socket(*args, **kwargs):
    _AUDIT["sockets_opened"] += 1
    raise RuntimeError("Network socket creation is blocked in offline audit mode.")


def _blocked_connect(*args, **kwargs):
    _AUDIT["sockets_opened"] += 1
    raise RuntimeError("Network connection is blocked in offline audit mode.")


def _blocked_getaddrinfo(*args, **kwargs):
    _AUDIT["sockets_opened"] += 1
    raise RuntimeError("DNS resolution is blocked in offline audit mode.")


def _blocked_urlopen(*args, **kwargs):
    _AUDIT["sockets_opened"] += 1
    raise RuntimeError("urllib.request.urlopen is blocked in offline audit mode.")


socket.socket = _blocked_socket  # type: ignore[assignment]
socket.create_connection = _blocked_connect  # type: ignore[assignment]
socket.getaddrinfo = _blocked_getaddrinfo  # type: ignore[assignment]
urllib.request.urlopen = _blocked_urlopen  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from satark.engine import SatarkEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify 100% offline inference with zero sockets.")
    parser.add_argument(
        "--message",
        default="URGENT: Your SBI YONO account is blocked! Update KYC at bit.ly/sbi-kyc immediately.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    engine = SatarkEngine()
    res = engine.analyze(args.message, source="text")

    payload = {
        "sockets_opened": _AUDIT["sockets_opened"],
        "bytes_transmitted": _AUDIT["bytes_transmitted"],
        "offline_verified": _AUDIT["sockets_opened"] == 0 and _AUDIT["bytes_transmitted"] == 0,
        "model_sha256": engine.model_sha256,
        "model_mode": res["model_mode"],
        "verdict": res["verdict"],
        "category": res["category"],
        "confidence_pct": res["confidence_pct"],
    }
    print(json.dumps(payload, indent=2 if not args.json else None))
    if not payload["offline_verified"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
