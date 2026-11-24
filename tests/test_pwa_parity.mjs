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

// Ask Python Lite to score 100 diverse samples from test_unseen.csv (or built-in fallback) + edge cases
const pyScript = `
import json, os, sys
from pathlib import Path
sys.path.insert(0, r"${ROOT.replace(/\\/g, "\\\\")}\\src")
from satark.engine import SatarkEngine

samples = []
csv_path = Path(os.environ.get("SATARK_DATA_DIR", r"C:\\Users\\arsal\\Desktop\\Projects\\TinyVerse\\data")) / "processed" / "test_unseen.csv"
try:
    import pandas as pd
    df = pd.read_csv(csv_path)
    samples = df.sample(n=min(88, len(df)), random_state=42)["text"].astype(str).tolist()
except Exception:
    samples = [
        "Dear customer, your HDFC Bank account will be blocked today. Update KYC at bit.ly/hdfc-rekyc",
        "URGENT: Your SBI YONO netbanking is suspended! Verify PAN at sbi-kyc-update.xyz",
        "Congratulations! You won Rs 25 Lakh in KBC Lucky Draw! Call 9876543210",
        "Pre-approved Instant Personal Loan of Rs 5,00,000! Zero CIBIL check: fast-loan-india.loan",
        "Work From Home! Earn Rs 3,500 daily liking YouTube videos. Message HR on t.me/daily_tasks",
        "CBI Cyber Cell: Digital arrest warrant issued for FedEx narcotics parcel. Call 9876543210",
        "Traffic e-Challan Notice: Vehicle KA03MN4821 fined Rs 2,500. Download RTO_Challan.apk at bit.ly/echallan-apk",
        "BESCOM Warning: Your electricity will be cut tonight at 9:30 PM. Call officer 9871234560",
        "Bro quick favor, send Rs 8000 to 9811223344@paytm right now, will return in 1 hour",
        "Sir I accidentally transferred Rs 15000 to your Google Pay. Please refund to 9876543210@ybl",
        "VM-HDFCBK-S: Rs 1,450.00 debited from A/c XX4821 on 23-11-26 via UPI. Avail Bal: Rs 24,590. hdfcbank.com",
        "AD-UIDAI-G: OTP for Aadhaar authentication is 849201. Never share OTP with anyone.",
        "AM-AMAZON-T: Out for delivery: Your Amazon order #402-123456 will arrive today by 8 PM. amazon.in",
        "Hi Arya, see the latest watches at Titan World, https://m.tneu.in/TITNWO/JeOf3mn Visit today.",
        "Today only! Double your KFC bucket. Buy 6 pcs Hot n Crispy and get 6 pcs Strips FREE! bit.ly/kfcind-dsg19 TnC",
        "HDFC ERGO: Your motor insurance policy expires on 15 Dec 2026. Renew online on hdfcergo.com",
        "URGENT: The production server is down and the client demo starts in 30 minutes! Join Zoom immediately.",
        "Hey sending you my UPI id for the Goa trip split: arsalan.dev@okaxis, total comes to Rs 3,400.",
        "Hello, how are you doing today?",
        "Thanks!",
    ]

samples.extend([
    "VM-SBIKYC: Aapka SBI YONO khata aaj band ho jayega! Update at bit.ly/sbi-kyc",
    "CONGRATULATIONS! Won Rs 25000! upi://pay?pa=win@ybl&pn=Prize&am=1999&cu=INR cashback",
    "VM-HDFCBK-S: Rs 1450 debited from A/c XX4821 via UPI Ref 123456. Avail Bal Rs 9000. hdfcbank.com",
    "Bhai 500 UPI kar de kal deta hu",
    "Claim ₹50k bit.ly/x",
    "Dear SBI Customer, update your PAN details at https://forms.gle/sbiKyc2026",
    "Email us at support@hdfcbank.com regarding vehicle KA03MN4821",
    "Bro send Rs 8000 to 9811223344@paytm right now",
    "Hey bro did you check your HDFC bank statement yet?",
    "🎉🔥🚨",
    "Urgent l04n offer from b4nk k.y.c at bit.ly/loan-now",
    "482910 is your verification code for Zepto. Valid for 5 minutes.",
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
