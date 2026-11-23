"""FastAPI server for Satark — serves compiled frontend/dist/ and offline inference APIs."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from satark.clipboard import ClipboardGuardian  # noqa: E402
from satark.engine import SatarkEngine  # noqa: E402
from satark.mutator import mutate_message  # noqa: E402

engine = SatarkEngine()
clipboard = ClipboardGuardian(poll_interval=0.4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    clipboard.start(loop)
    yield
    clipboard.stop()


app = FastAPI(title="Satark — On-Device Scam Detector", version="1.0.0", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    source: Literal["text", "clipboard", "share_target", "ocr_future"] = "text"

    @field_validator("text")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty or whitespace-only")
        return v


class MutateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    level: Literal["light", "medium", "hard"] = "medium"

    @field_validator("text")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty or whitespace-only")
        return v


class ClipboardToggleRequest(BaseModel):
    enabled: bool


class OfflineAuditRequest(BaseModel):
    text: str = Field(
        default="URGENT: Your SBI YONO account is blocked! Update KYC at bit.ly/sbi-kyc immediately.",
        min_length=1,
        max_length=10_000,
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_mode": engine.mode,
        "model_sha256": engine.model_sha256,
        "clipboard_supported": clipboard.supported,
        "clipboard_enabled": clipboard.enabled,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    return engine.analyze(req.text, source=req.source)


@app.post("/api/mutate")
def mutate(req: MutateRequest) -> dict:
    orig_res = engine.analyze(req.text, source="text")
    mutated_text = mutate_message(req.text, level=req.level)
    mut_res = engine.analyze(mutated_text, source="text")
    delta = round(mut_res["confidence_pct"] - orig_res["confidence_pct"], 1)
    return {
        "original": orig_res,
        "mutated": mut_res,
        "mutated_text": mutated_text,
        "confidence_delta": delta,
    }


@app.post("/api/clipboard/toggle")
def toggle_clipboard(req: ClipboardToggleRequest) -> dict:
    enabled = clipboard.set_enabled(req.enabled)
    return {"enabled": enabled, "supported": clipboard.supported}


@app.get("/api/clipboard/stream")
async def clipboard_stream() -> StreamingResponse:
    q = clipboard.subscribe()

    async def event_generator():
        try:
            yield f"event: status\ndata: {json.dumps({'enabled': clipboard.enabled, 'supported': clipboard.supported})}\n\n"
            while True:
                try:
                    text = await asyncio.wait_for(q.get(), timeout=15.0)
                    result = await asyncio.to_thread(engine.analyze, text, "clipboard")
                    yield f"event: scan\ndata: {json.dumps(result)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            clipboard.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/audit/offline")
def audit_offline(req: OfflineAuditRequest) -> dict:
    script_path = ROOT / "tests" / "verify_offline.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail="verify_offline.py not found")
    proc = subprocess.run(
        [sys.executable, str(script_path), "--json", "--message", req.text],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Offline audit failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout.strip())


@app.get("/api/benchmarks")
def benchmarks() -> dict:
    summary_path = ROOT / "models" / "lite" / "benchmarks_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="benchmarks_summary.json not found")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["live_model_mode"] = engine.mode
    data["live_model_sha256"] = engine.model_sha256
    return data


DIST_DIR = ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
