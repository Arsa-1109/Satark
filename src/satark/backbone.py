"""Layer 2: Frozen Multilingual-MiniLM INT8 ONNX Encoder (SDD §2.3)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class BackboneNotAvailableError(RuntimeError):
    """Raised when ONNX backbone or tokenizer files are missing."""


class OnnxBackbone:
    """CPU-native INT8 MiniLM-L12 encoder using onnxruntime + tokenizers."""

    def __init__(self, onnx_dir: Path, max_length: int = 128) -> None:
        self.onnx_dir = Path(onnx_dir)
        self.model_path = self.onnx_dir / "model_quantized.onnx"
        self.tok_path = self.onnx_dir / "tokenizer.json"
        if not self.model_path.exists() or not self.tok_path.exists():
            raise BackboneNotAvailableError(
                f"ONNX backbone not found in {self.onnx_dir}. Run scripts/fetch_backbone.py."
            )

        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(self.tok_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(length=None)
        self._sha256: str | None = None

    def get_sha256(self) -> str:
        if self._sha256 is None:
            h = hashlib.sha256()
            with self.model_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            self._sha256 = h.hexdigest()
        return self._sha256

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return L2-normalised 384-d float32 mean-pooled embeddings."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        all_embs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encodings = self.tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            feeds: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

            outputs = self.session.run(None, feeds)
            token_embs = outputs[0]  # (B, L, 384)
            mask = attention_mask[:, :, None].astype(np.float32)
            summed = (token_embs * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = summed / counts
            norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), a_min=1e-12, a_max=None)
            all_embs.append((pooled / norms).astype(np.float32))

        return np.vstack(all_embs)
