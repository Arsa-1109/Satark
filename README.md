# Satark — 100% Offline, On-Device Indian Scam Message Detector

**TinyVerse Finals — Problem Statement 1 (`TV004`)**

Satark detects Indian SMS and WhatsApp scams **100% offline on CPU** across both Laptop (`FastAPI` + `ONNX Runtime`) and Mobile (`Offline PWA` via `Service Worker`), returning a calibrated `SCAM`/`SAFE` verdict, 5-class category (`KYC/Bank`, `Lottery/Prize`, `Loan/Credit`, `Job/Investment`, `Other`), verdict certainty, 4-Pillar Social-Engineering X-Ray, `upi://pay` payload decoding, and a PII-redacted 1930/Chakshu report draft.

---

## Architecture

1. **Layer 1 — Adversarial & Indic Phonetic Normaliser (`src/satark/normalizer.py`) + 35 Indian Forensics (`src/satark/forensics.py`)**
   - Unicode `NFKC`, zero-width/bidi stripping, Cyrillic/Greek homoglyph folding, spaced-token collapsing (`K.Y.C` → `kyc`), token-level leetspeak decoding (`l04n` → `loan`), and Indic Hinglish phonetic canonicalisation.
   - Exact 35-feature vector `f[0..34]` covering TRAI DLT headers, `.bank.in` / official Indian domains, brand-domain mismatch, `upi://pay` collect/cashback traps, `.apk` sideloading lures, AnyDesk/TeamViewer, Punycode (`xn--`), and anti-fraud OTP warnings.
2. **Layer 2 — Dual-Mode Calibrated ML Engine (`src/satark/backbone.py`, `src/satark/engine.py`)**
   - **Full ONNX Hybrid (`full_onnx_hybrid`)**: Frozen `paraphrase-multilingual-MiniLM-L12-v2` INT8 ONNX (`max_length=128`, 384-d mean-pooled) + Char(3–5)/Word(1–2) TF-IDF + 35 scaled forensics → Platt-calibrated Logistic Regression + LightGBM stacking + Shannon-entropy OOD routing to `Other`.
   - **Lite Fallback (`lite_fallback`, `0.22 MB` in `models/lite/`)**: Pure Platt-calibrated Logistic Regression on TF-IDF + 35 forensics; runs immediately after `git clone` with zero downloads.
   - **Offline Mobile PWA (`pwa_js`, `0.54 MB` `frontend/public/model_bundle.js`)**: Exact 1-to-1 JS port (`frontend/src/engine/localEngine.js`) with Android `share_target` and Service Worker caching.
3. **Layer 3 — Swiss Editorial UI (`frontend/`) + Windows Clipboard Guardian (`src/satark/clipboard.py`) + CLI (`cli.py`)**

---

## Verified Quantitative Results (Gates G1–G14)

| Gate | Metric | Full ONNX Hybrid | Lite Fallback / PWA | Target |
| :--- | :--- | :---: | :---: | :---: |
| **G1** | Unseen Binary F1 (`test_unseen.csv`) | **0.9755** | **0.9654** | $\ge 0.94$ |
| **G2** | Hard-Negative TRAI DLT Safe FPR (`test_hard_negatives.csv`) | **0.0%** (0/300) | **0.0%** (0/300) | $\le 2.0\%$ |
| **G3** | 5-Class Category Macro-F1 (`test_unseen.csv`) | **1.0000** | **1.0000** | $\ge 0.88$ |
| **G4** | Adversarial & Obfuscated Recall (`test_adversarial.csv`) | **1.0000** | **1.0000** | $\ge 0.90$ |
| **G5** | Calibration (`ECE` / `Brier Score`) | **0.0045** / **0.0056** | **0.0038** / **0.0084** | $\le 0.06$ / $\le 0.08$ |
| **G6** | Median CPU Latency | **34.48 ms** | **5.18 ms** | $< 80\text{ ms}$ / $< 10\text{ ms}$ |
| **G7** | Python Lite vs PWA JS Parity (100 samples) | — | **max $\Delta = 0.0001$**, 0 mismatches | $< 0.005$ |
| **G8** | Subprocess Zero-Network Audit (`tests/verify_offline.py`) | **0 sockets, 0 bytes** | **0 sockets, 0 bytes** | `0` sockets / `0` bytes |
| **G9** | GitHub Packaging (`< 100 MB` files, zero CDN in `frontend/dist/`) | **Pass** | **Pass** | Pass |
| **G10** | Held-Out Real-World Indian Stress (`test_realworld_stress.csv`, 50 msgs) | **100.0%** (50/50) | **100.0%** (50/50) | $\ge 92.0\%$ |
| **G11** | Blind OOD Safe FPR (`test_ood_safe.csv`, 31 msgs) | **0.0%** (0/31) | **3.23%** (1/31) | $\le 5.0\%$ |
| **G12** | Headerless Transactional & Promo FPR (`test_transactional_safe.csv`, 25 msgs) | **0.0%** (0/25) | **0.0%** (0/25) | $\le 4.0\%$ |
| **G13** | Blind Multilingual + Novel Scam + Tricky Safe (`test_blind_multilingual.csv`, 24 msgs) | **100.0%** (24/24) | **100.0%** (24/24) | $\ge 92.0\%$ |
| **G14** | Blind Commercial Ads Safe FPR (`test_ads_safe.csv`, 20 msgs) | **5.0%** (1/20) | **0.0%** (0/20) | $\le 5.0\%$ |

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Fetch 118 MB INT8 ONNX Backbone for `full_onnx_hybrid` Mode
> Without this step, Satark runs immediately in `lite_fallback` mode using the committed `0.22 MB` weights in `models/lite/`.
```bash
python scripts/fetch_backbone.py
```

### 3. Run Laptop UI & API (`http://127.0.0.1:8000`)
```bash
run.bat
# or:
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### 4. CLI Usage
```bash
python cli.py "VM-SBIKYC: Aapka SBI khata band hoga! Update at bit.ly/sbi-kyc"
python cli.py "VM-SBIKYC: Aapka SBI khata band hoga! Update at bit.ly/sbi-kyc" --json
```

### 5. Airplane-Mode & Verification Suite
```bash
python tests/verify_offline.py --json
node tests/test_pwa_parity.mjs
pytest tests/test_edge_cases.py -v
```

---

## Repository Layout

- `app.py`, `cli.py`, `run.bat`, `requirements.txt`
- `src/satark/`: `normalizer.py`, `forensics.py`, `backbone.py`, `engine.py`, `mutator.py`, `redact.py`, `clipboard.py`
- `scripts/`: `fetch_backbone.py`, `build_dataset.py`, `train.py`, `export_pwa.py`, `evaluate.py`
- `models/lite/`: committed `< 4 MB` Lite Fallback weights + `benchmarks_summary.json`
- `models/onnx/`: gitignored 118 MB INT8 MiniLM ONNX backbone + full heads
- `frontend/`: React 19 + Vite 6 + Tailwind CSS v4 + Offline PWA (`public/model_bundle.js`, `public/manifest.json`, `public/sw.js`, `dist/`)
- `tests/`: `verify_offline.py`, `test_pwa_parity.mjs`, `test_edge_cases.py`

---

## Attribution & Credits

- **Datasets**:
  - UCI SMS Spam Collection (Almeida & Gómez Hidalgo, 2012) — weighted `0.35` for Stage A binary only.
  - Hugging Face Indian/Hinglish SMS datasets: `CloveAI/india-spam-sms`, `anmolshrivastav/scam-ham-india`.
  - Curated 2025–26 Indian 5-class scam & TRAI DLT hard-negative evaluation suite.
- **Pretrained Model**:
  - `onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX` (`model_quantized.onnx`, Apache-2.0 / Sentence-Transformers).
- **Libraries**:
  - `onnxruntime`, `tokenizers`, `scikit-learn`, `lightgbm`, `fastapi`, `uvicorn`, `huggingface_hub`, `numpy`, `scipy`, `joblib`, `pandas`, `pydantic`, `pytest`, `httpx`, `react`, `vite`, `tailwindcss`, `lucide-react`.
- **AI Assistance**:
  - Built with AI-assisted engineering (Antigravity) for architecture refinement, dataset curation, JS parity implementation, and automated test verification.
