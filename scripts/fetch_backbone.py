"""Fetch and SHA-256 verify the frozen Multilingual-MiniLM INT8 ONNX backbone."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "models" / "onnx"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model_dest = ONNX_DIR / "model_quantized.onnx"
    tok_dest = ONNX_DIR / "tokenizer.json"

    if not model_dest.exists():
        print(f"Downloading onnx/model_quantized.onnx from {REPO_ID}...")
        cached = Path(hf_hub_download(repo_id=REPO_ID, filename="onnx/model_quantized.onnx"))
        shutil.copy2(cached, model_dest)
    else:
        print(f"Found existing {model_dest}")

    if not tok_dest.exists():
        print(f"Downloading tokenizer.json from {REPO_ID}...")
        cached = Path(hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json"))
        shutil.copy2(cached, tok_dest)
    else:
        print(f"Found existing {tok_dest}")

    manifest = {
        "repo_id": REPO_ID,
        "max_length": 128,
        "model_file": "model_quantized.onnx",
        "model_size_bytes": model_dest.stat().st_size,
        "model_sha256": sha256_file(model_dest),
        "tokenizer_file": "tokenizer.json",
        "tokenizer_size_bytes": tok_dest.stat().st_size,
        "tokenizer_sha256": sha256_file(tok_dest),
    }
    manifest_path = ONNX_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
