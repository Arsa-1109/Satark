/**
 * Gate G7: Python Lite vs React PWA JS Parity Test across 100 samples.
 * Asserts max(|p_py - p_js|) < 0.005 and 100% verdict/category match.
 */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

import { analyzeLocal } from "../frontend/src/engine/localEngine.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// Load window.__SATARK_MODEL__ via vm sandbox
const bundleCode = fs.readFileSync(path.join(ROOT, "frontend", "public", "model_bundle.js"), "utf-8");
const sandbox = { window: {} };
vm.runInNewContext(bundleCode, sandbox);
const modelBundle = sandbox.window.__SATARK_MODEL__;

// Ask Python Lite to score 100 diverse samples from test_unseen.csv + edge cases
const pyScript = `
import json, sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"${ROOT.replace(/\\/g, "\\\\")}\\src")
from satark.engine import SatarkEngine

df = pd.read_csv(r"C:\\Users\\arsal\\Desktop\\Projects\\TinyVerse\\data\\processed\\test_unseen.csv")
samples = df.sample(n=min(94, len(df)), random_state=42)["text"].astype(str).tolist()
samples.extend([
    "VM-SBIKYC: Aapka SBI YONO khata aaj band ho jayega! Update at bit.ly/sbi-kyc",
    "CONGRATULATIONS! Won Rs 25000! upi://pay?pa=win@ybl&pn=Prize&am=1999&cu=INR cashback",
    "VM-HDFCBK-S: Rs 1450 debited from A/c XX4821 via UPI Ref 123456. Avail Bal Rs 9000. hdfcbank.com",
    "Bhai 500 UPI kar de kal deta hu",
    "Claim ₹50k bit.ly/x",
    "🎉🔥🚨",
])
eng = SatarkEngine(force_lite=True)
out = []
for s in samples[:100]:
    r = eng.analyze(s, force_lite=True)
    out.append({"text": s, "p": r["scam_probability"], "verdict": r["verdict"], "category": r["category"]})
print(json.dumps(out))
`;

const pyJson = execFileSync("python", ["-c", pyScript], { encoding: "utf-8", maxBuffer: 20 * 1024 * 1024 });
const pyRows = JSON.parse(pyJson);

let maxDelta = 0;
let verdictMismatches = 0;
let catMismatches = 0;

for (let i = 0; i < pyRows.length; i++) {
  const row = pyRows[i];
  const jsRes = analyzeLocal(row.text, "text", modelBundle);
  const delta = Math.abs(row.p - jsRes.scam_probability);
  if (delta > maxDelta) maxDelta = delta;
  if (row.verdict !== jsRes.verdict) {
    verdictMismatches++;
    console.error(`Verdict mismatch #${i}: py=${row.verdict} (${row.p}) vs js=${jsRes.verdict} (${jsRes.scam_probability})`);
  }
  if (row.category !== jsRes.category) {
    catMismatches++;
    console.error(`Category mismatch #${i}: py=${row.category} vs js=${jsRes.category}`);
  }
}

console.log(
  JSON.stringify(
    {
      samples_tested: pyRows.length,
      max_prob_delta: Number(maxDelta.toFixed(6)),
      verdict_mismatches: verdictMismatches,
      category_mismatches: catMismatches,
      gate_g7_passed: maxDelta < 0.005 && verdictMismatches === 0 && catMismatches === 0,
    },
    null,
    2
  )
);

if (maxDelta >= 0.005 || verdictMismatches > 0 || catMismatches > 0) {
  process.exit(1);
}
