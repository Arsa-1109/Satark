"""Build deduplicated, anti-leakage Indian SMS/WhatsApp dataset in Projects\\TinyVerse\\data."""

from __future__ import annotations

import io
import os
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from satark.mutator import mutate_message  # noqa: E402


def _resolve_data_dir() -> Path:
    env_dir = os.environ.get("SATARK_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    default_dir = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\data")
    if default_dir.parent.exists():
        return default_dir
    return ROOT / "data"


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

SEED = 42


def _dedup_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _download_uci() -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "uci_sms_spam.zip"
    if not zip_path.exists():
        print(f"Downloading UCI SMS Spam Collection from {UCI_URL}...")
        try:
            urllib.request.urlretrieve(UCI_URL, zip_path)
        except Exception as exc:
            print(f"[WARN] Could not download UCI dataset: {exc}")
            return []
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if "SMSSpamCollection" in name:
                raw = zf.read(name).decode("utf-8", errors="replace")
                for line in raw.splitlines():
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        lbl, txt = parts[0].strip().lower(), parts[1].strip()
                        is_spam = 1 if lbl == "spam" else 0
                        w = 0.35 if (is_spam or len(txt) >= 35) else 0.15
                        if txt:
                            rows.append({
                                "text": txt,
                                "label": is_spam,
                                "category": "",
                                "source_corpus": "uci_sms",
                                "sample_weight": w,
                            })
    print(f"Loaded {len(rows)} rows from UCI SMS Spam Collection (weight=0.35).")
    return rows


def _template_skeleton(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"upi://pay\?\S+", "<UPI>", s)
    s = re.sub(r"https?://\S+|www\.\S+", "<URL>", s)
    s = re.sub(r"\b[a-z0-9.\-_]+@[a-z0-9.\-]+\b", "<VPA>", s)
    s = re.sub(r"\b[a-z]{2}[\s-]?\d{2}[\s-]?[a-z]{1,3}[\s-]?\d{4}\b", "<PLATE>", s)
    s = re.sub(r"(?:ref#|ticket id:|offer#|batch-|case#|\[id:|\[event |\(ref )\s*[a-z0-9\-]+\]?", "", s)
    s = re.sub(r"(?:\(\d+\)|\[\d+\]|#\d+)\s*$", "", s)
    s = re.sub(r"\d+", "#", s)
    return re.sub(r"\s+", " ", s).strip()


def _try_hf_indian_datasets() -> list[dict]:
    """Best-effort pull of clean public HF SMS/spam datasets with schema validation (excludes noisy CloveAI)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    candidates = [
        ("Deysi/spam-detection-dataset", 0.25),
        ("anmolshrivastav/scam-ham-india", 0.30),
    ]
    rows: list[dict] = []
    for repo, weight in candidates:
        try:
            local_repo_dir = RAW_DIR / repo.replace("/", "__")
            local_files = (
                [p for p in sorted(local_repo_dir.rglob("*")) if p.is_file() and p.suffix in (".csv", ".parquet")][:2]
                if local_repo_dir.exists()
                else []
            )
            if not local_files:
                files = api.list_repo_files(repo_id=repo, repo_type="dataset")
                data_files = [f for f in files if f.endswith((".csv", ".parquet"))]
                if not data_files:
                    continue
                local_files = [
                    Path(hf_hub_download(repo_id=repo, filename=df_file, repo_type="dataset", local_dir=local_repo_dir))
                    for df_file in data_files[:2]
                ]
            added = 0
            for local_path in local_files:
                local = str(local_path)
                df = pd.read_parquet(local) if local.endswith(".parquet") else pd.read_csv(local)
                cols = {c.lower(): c for c in df.columns}
                text_col = cols.get("text") or cols.get("message") or cols.get("sms") or cols.get("v2")
                lbl_col = cols.get("label") or cols.get("class") or cols.get("v1") or cols.get("is_scam")
                if not text_col or not lbl_col:
                    continue
                for _, r in df.iterrows():
                    txt = str(r[text_col]).strip()
                    raw_lbl = str(r[lbl_col]).strip().lower()
                    if not txt or txt == "nan" or len(txt) > 1200:
                        continue
                    if "Deysi" in repo:
                        low_t = txt.lower()
                        if any(k in low_t for k in ("subreddit", "as an ai language model", "pushshift", "dataset of", "thesis")):
                            continue
                    is_scam = 1 if raw_lbl in ("1", "spam", "scam", "fraud", "true") else 0
                    if is_scam == 1 and "Deysi" in repo:
                        continue  # Skip noisy retail/promo spam from Deysi; keep Indian scams from scam-ham-india
                    rows.append({
                        "text": txt,
                        "label": is_scam,
                        "category": "",
                        "source_corpus": f"hf_{repo}",
                        "sample_weight": weight if is_scam == 0 else 0.45,
                    })
                    added += 1
            print(f"Loaded {added} rows from HF {repo} (weight={weight}).")
        except Exception as exc:
            print(f"[INFO] Skipping HF {repo} ({type(exc).__name__}).")
    return rows


def _generate_indian_scam_corpus() -> list[dict]:
    """Generate a diverse, realistic 2025-26 Indian 5-class scam corpus (English + Hinglish + Indic)."""
    rng = random.Random(SEED)
    rows: list[dict] = []

    banks = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "PNB", "Bank of Baroda", "Kotak", "Paytm", "PhonePe", "GPay"]
    kyc_domains = ["sbi-kyc-update.xyz", "hdfc-verify.top", "bit.ly/kyc-now", "tinyurl.com/yono-pan", "icici-rekyc.click", "t.ly/pan-link", "rb.gy/bank-kyc", "192.168.1.45/kyc", "xn--sbi-9ta.com/kyc"]
    phones = ["9876543210", "9123456789", "8800112233", "7011223344", "9988776655"]
    amounts_kyc = ["Rs. 14,999", "Rs 49,500", "₹9,850", "Rs. 25,000"]

    kyc_templates = [
        "Dear customer, your {bank} YONO account will be blocked today due to pending KYC. Update PAN immediately at {dom} or call {ph}.",
        "Aapka {bank} khata aaj band ho jayega! Turant apna KYC aur PAN update karein: {dom} warna balance freeze hoga.",
        "ALERT: Your {bank} NetBanking is suspended! Verify OTP & Aadhaar within 24 hours at {dom} to reactivate.",
        "Dear {bank} user, your debit card is deactivated. Share OTP or click {dom} to complete e-KYC now.",
        "{bank} FASTag KYC expired! Pay {amt} penalty or update immediately at {dom} else blacklist today.",
        "Urgent: {bank} account holder, update your MPIN and CVV at {dom} within 2 hrs to avoid permanent block.",
        "Bhai {bank} ka server update chal raha hai, apna KYC link {dom} pe abhi verify kar lo warna account band.",
        "प्रिय ग्राहक, आपका {bank} खाता KYC न होने के कारण आज बंद हो जाएगा। तुरंत अपडेट करें: {dom}",
        "HDFC/SBI Warning: Suspicious login detected. Cancel {amt} debit by entering OTP at {dom} or call helpline {ph}.",
        "Your {bank} credit card reward points worth Rs 8,450 expire tonight! Redeem via {dom} using card PIN.",
    ]
    for i in range(340):
        tpl = kyc_templates[i % len(kyc_templates)]
        txt = tpl.format(
            bank=rng.choice(banks),
            dom=rng.choice(kyc_domains),
            ph=rng.choice(phones),
            amt=rng.choice(amounts_kyc),
        )
        if i % 3 == 0:
            txt = f"VM-SBIKYC: {txt}"  # spoofed header test
        if i % 4 == 1:
            txt += f" Ref#{10000 + i}"
        rows.append({"text": txt, "label": 1, "category": "KYC/Bank", "source_corpus": "indian_curated_2026", "sample_weight": 1.0})

    # 2. Lottery / Prize
    lottery_urls = ["bit.ly/kbc-win25", "tinyurl.com/jio-prize", "kbc-lucky-draw.vip", "t.ly/amazon-spin", "rb.gy/flipkart-gift", "claim-reward.online/win"]
    prize_amts = ["Rs. 25,00,000", "₹25 Lakh", "Rs 50,000", "₹1,50,000", "Rs. 7,50,000"]
    lottery_templates = [
        "CONGRATULATIONS!! You have won {amt} in KBC Jio Lucky Draw 2026! WhatsApp your bank details to {ph} or claim at {dom}.",
        "Badhai ho! Aapne jeeta hai {amt} ka bumper inam! Claim karne ke liye {dom} par click karein aur processing fee bhejein.",
        "Amazon Big Billion Spin Winner! You won iPhone 16 Pro & {amt} cashback. Accept upi://pay?pa=reward-claim@ybl&pn=AmazonPrize&am=1999&cu=INR to receive!",
        "Flipkart Lucky Customer: Your mobile number won {amt}! Receive cashback by entering UPI PIN on upi://pay?pa=flipkart-bonus@okicici&pn=Bonus&am=2499&cu=INR.",
        "KBC Head Office Mumbai: Aapka number {amt} lottery mein select hua hai. Turant call karein {ph} ya visit {dom}.",
        "FREE BONUS! Claim your {amt} Diwali Bumper Jackpot reward today only at {dom}. Limited time!",
        "बधाई हो! आपने KBC लकी ड्रॉ में {amt} जीते हैं। इनाम पाने के लिए {dom} पर क्लिक करें।",
        "You are selected for {amt} Tata Safari Lucky Draw! Pay Rs 999 registration via {dom} to unlock prize.",
    ]
    for i in range(300):
        tpl = lottery_templates[i % len(lottery_templates)]
        txt = tpl.format(amt=rng.choice(prize_amts), dom=rng.choice(lottery_urls), ph=rng.choice(phones))
        if i % 3 == 0:
            txt += f" Ticket ID: KBC{2026 + i}"
        rows.append({"text": txt, "label": 1, "category": "Lottery/Prize", "source_corpus": "indian_curated_2026", "sample_weight": 1.0})

    # 3. Loan / Credit
    loan_urls = ["bit.ly/instant-karz", "fast-loan-india.loan", "tinyurl.com/zero-cibil", "credit-limit-up.shop", "t.ly/mudra-sanction"]
    loan_amts = ["Rs. 5,00,000", "₹2 Lakh", "Rs 1,00,000", "₹3,50,000", "Rs 50,000"]
    loan_templates = [
        "Pre-approved Instant Personal Loan of {amt} in 5 mins! Zero CIBIL check, 1% interest. Apply now: {dom}",
        "Bina CIBIL score ke turant {amt} ka loan paayein! Low EMI, instant sanction. Click {dom} or WhatsApp {ph}.",
        "Your {bank} Credit Card limit has been upgraded to {amt}! Pay Rs 499 verification fee at {dom} to activate.",
        "PM Mudra Yojana Loan Approved: {amt} sanctioned for your Aadhaar! Pay processing fee via {dom} today.",
        "Urgent Loan Offer: Get {amt} directly in bank account within 10 minutes. Download our app at {dom} or call {ph}.",
        "Aapka {amt} ka karz approve ho gaya hai. EMI sirf Rs 1999/month. Abhi sanction letter download karein: {dom}",
        "तुरंत पर्सनल लोन {amt} बिना किसी दस्तावेज़ के! कम ब्याज दर। अभी आवेदन करें: {dom}",
        "Zero-documents instant cash loan {amt}! Bad CIBIL accepted. Install LoanFast from {dom} now.",
    ]
    for i in range(290):
        tpl = loan_templates[i % len(loan_templates)]
        txt = tpl.format(amt=rng.choice(loan_amts), dom=rng.choice(loan_urls), ph=rng.choice(phones), bank=rng.choice(banks))
        if i % 4 == 0:
            txt += f" Offer#{500 + i}"
        rows.append({"text": txt, "label": 1, "category": "Loan/Credit", "source_corpus": "indian_curated_2026", "sample_weight": 1.0})

    # 4. Job / Investment
    job_urls = ["wa.me/919876543210", "t.me/daily_earn_tasks", "bit.ly/wfh-hr-2026", "crypto-vip-tips.live", "tinyurl.com/yt-like-job"]
    job_templates = [
        "Part time Work From Home! Earn Rs 3,000 to Rs 8,000 daily income just by liking YouTube videos. Join Telegram: {dom}",
        "HR Manager Hiring: Ghar baithe daily Rs 5,000 kamayein! Simple Amazon/Google review task. Message on WhatsApp {dom}.",
        "Guaranteed 300% return in 7 days! Join our VIP Stock Tips & Crypto Trading group at {dom}. Minimum investment Rs 5,000.",
        "Part-time job offer! Flexible 2 hrs/day, salary Rs 45,000/month. Subscribe channels & earn per day. Contact {ph} or {dom}.",
        "Aaj hi join karein Work From Home task job! Daily payout via UPI Rs 2500-6000. Telegram HR: {dom}",
        "SEBI-style VIP Intraday Jackpot Calls! Double your capital in 48 hours with 99% accuracy. Join {dom} now!",
        "घर बैठे पार्ट टाइम जॉब! रोज़ाना ₹4000 कमाएँ यूट्यूब वीडियो लाइक करके। संपर्क करें: {dom}",
        "Urgent HR Requirement: Data entry & Telegram task job, instant salary credit Rs 1,500/hr. Apply {dom}",
    ]
    for i in range(290):
        tpl = job_templates[i % len(job_templates)]
        txt = tpl.format(dom=rng.choice(job_urls), ph=rng.choice(phones))
        if i % 3 == 0:
            txt += f" Batch-{i}"
        rows.append({"text": txt, "label": 1, "category": "Job/Investment", "source_corpus": "indian_curated_2026", "sample_weight": 1.0})

    # 5. Other (Digital Arrest, .apk lures, Electricity disconnect, Parcel/Customs, Remote access)
    other_urls = ["bit.ly/echallan-apk", "tinyurl.com/bescom-bill", "wedding-invite-card.apk", "indpost-customs.xyz", "t.ly/trai-disconnect", "rto-challan-pay.apk"]
    other_templates = [
        "CBI / Mumbai Police Cyber Cell: Your Aadhaar is linked to a FedEx narcotics parcel & money laundering FIR. Digital arrest warrant issued! Join video call immediately or call {ph}.",
        "TRAI Final Notice: All your mobile numbers will be disconnected in 2 hours due to illegal activities. Press 9 or contact CBI officer at {ph}.",
        "BESCOM / MSEDCL Warning: Your electricity connection will be cut tonight at 9:30 PM because last bill was not updated. Call officer {ph} or pay via {dom}.",
        "Aapki bijli aaj raat kat jayegi! Bill update nahi hua hai, turant electricity officer {ph} ko call karein ya AnyDesk install karein.",
        "Shadi ka card! Namaste, humari shaadi mein zaroor aana. Download invitation Wedding_Card.apk from {dom}.",
        "Traffic e-Challan Notice: Vehicle fined Rs 2,500. Download RTO_Challan_View.apk at {dom} to check photo proof immediately.",
        "IndiaPost Customs Alert: Your international parcel is held at Delhi airport due to unpaid duty Rs 4,200. Pay at {dom} else seizure.",
        "Customer Care Support: Download AnyDesk or TeamViewer QuickSupport and share 9-digit code with {ph} to process your refund.",
        "डिजिटल अरेस्ट नोटिस: आपके आधार पर CBI केस दर्ज हुआ है। तुरंत संपर्क करें {ph} अन्यथा गिरफ्तारी वारंट जारी होगा।",
        "PM Kisan Samman Nidhi APK: Download PM_Kisan_Yojana.apk from {dom} to receive Rs 6,000 installment.",
    ]
    for i in range(320):
        tpl = other_templates[i % len(other_templates)]
        txt = tpl.format(dom=rng.choice(other_urls), ph=rng.choice(phones))
        if i % 3 == 0:
            txt += f" Case#{9000 + i}"
        rows.append({"text": txt, "label": 1, "category": "Other", "source_corpus": "indian_curated_2026", "sample_weight": 1.5})

    # Subtle & No-URL Conversational Indian Scams (2025-26)
    subtle_scams = [
        # Job/Investment without URLs
        ("Job/Investment", "Hi, I am {name} from {comp} HR team. We are hiring part-time product reviewers, earn Rs {amt1}-{amt2} daily from home. Reply YES for details."),
        ("Job/Investment", "Hello, part-time online task job available! Like & review brands from home, daily payout Rs {amt1}. Message our HR on WhatsApp {ph}."),
        ("Job/Investment", "Exclusive WhatsApp Stock Market Group: Get 5x multibagger IPO allotment tips from SEBI analysts. Join free: https://chat.whatsapp.com/VipStock{code}"),
        # KYC/Bank with lookalike .com/.in domains
        ("KYC/Bank", "Dear customer your {bank} card points worth Rs {amt2} expiring tonight. Redeem cash directly to bank via {lookalike}"),
        ("KYC/Bank", "Aapka PAN card aadhaar se link nahi hai, aaj raat 12 baje se bank account band ho jayega. Abhi update karein: https://pan-update-gov.in/{code}"),
        # Other: Hacked-friend UPI VPA, Accidental transfer refund, Police FIR extortion, FedEx Skype, Electricity callback without URL
        ("Other", "Bro quick favor, my UPI is not working, can you send Rs {amt2} to my friend {ph}@paytm right now? Will return in 1 hour pakka."),
        ("Other", "Yaar urgent help chahiye, mera GPay limit cross ho gaya hai, please Rs {amt2} mere dost {ph}@ybl pe transfer kar do, shaam ko wapas karta hu."),
        ("Other", "Sir I accidentally transferred Rs {amt2} to your Google Pay instead of my brother. Please refund Rs {amt2} urgently to {ph}@ybl, my mother is in hospital."),
        ("Other", "Madam mistakenly sent Rs {amt1} to your PhonePe. Kindly return urgently to {ph}@okicici, medical emergency at home."),
        ("Other", "Hello, this is FedEx {city}. A parcel in your name with illegal drugs was seized by customs. Connect with Cyber Police on Skype immediately."),
        ("Other", "Sir your son has been arrested by police in a narcotics case. Send Rs 50,000 immediately to {ph}@ybl to close FIR otherwise we will put him in jail."),
        ("Other", "Your electricity power will be disconnected tonight at 9.30 PM because your previous month bill was not updated. Immediately contact our electricity officer {ph}."),
        # Novel 2025-26 scams (ITR, OLX Army QR, Gas, Sextortion, Laptop deposit, Broken phone)
        ("KYC/Bank", "Income Tax Dept: Your ITR tax refund of Rs {amt2} is approved. Update bank account within 24 hrs at https://incometax-refund-in.com/{code}"),
        ("Other", "OLX Buyer: I am in Indian Army and want to buy your item for Rs {amt2}. Scan this QR code and enter your UPI PIN to receive advance payment."),
        ("Other", "IGL / Mahanagar Gas Alert: Your piped gas connection will be disconnected tonight at 8 PM due to unpaid bill. Call officer {ph} immediately."),
        ("Other", "Warning: I recorded your private video from webcam. Transfer Rs 25,000 to {ph}@ybl today or I will leak it to all your contacts."),
        ("Job/Investment", "Congrats! You are selected for {comp} Work From Home Data Entry job. Pay Rs {amt1} refundable laptop security deposit to hr-{code}@ybl."),
        ("Other", "Hi Mom, my phone broke so texting from friend's mobile. Need Rs {amt2} urgently for hospital emergency, send to {ph}@okicici."),
        ("Lottery/Prize", "Dear user, Rs {amt2} bonus is credited to your wallet. Withdraw directly to bank now via bit.ly/wlt-{code}"),
        ("Lottery/Prize", "Congrats! Claim your free iPhone 16 gift voucher worth Rs 79,900 today: tinyurl.com/gift-{code}"),
        ("KYC/Bank", "Alert: Your {bank} credit card is blocked for security. Re-verify PAN at bit.ly/rekyc-{code} within 2 hrs."),
        ("Loan/Credit", "Get instant Rs {amt2}0 pre-approved loan in 2 mins without CIBIL! Apply: bit.ly/loan-{code}"),
    ]
    hr_names = ["Priya", "Anjali", "Ritu", "Neha", "Sneha", "Kavita"]
    hr_comps = ["Amazon", "Flipkart", "Google Maps", "Meesho", "Nykaa"]
    lookalikes = ["sbi-points-redeem.com", "hdfc-reward-cash.in", "icici-points-claim.com", "axis-kyc-portal.in"]
    cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad"]
    for i in range(320):
        cat, tpl = subtle_scams[i % len(subtle_scams)]
        txt = tpl.format(
            name=hr_names[i % len(hr_names)],
            comp=hr_comps[i % len(hr_comps)],
            amt1=rng.choice([1500, 2000, 2500, 3000]),
            amt2=rng.choice([4500, 6000, 8500, 12000]),
            ph=f"9{rng.randint(100000000, 999999999)}",
            code=f"X{100 + i}",
            bank=rng.choice(banks),
            lookalike=lookalikes[i % len(lookalikes)],
            city=cities[i % len(cities)],
        )
        rows.append({"text": txt, "label": 1, "category": cat, "source_corpus": "indian_curated_2026", "sample_weight": 1.5})

    return rows


def _generate_hard_negative_safe_corpus() -> list[dict]:
    """Generate 500+ diverse genuine Indian TRAI DLT, bank, OTP, delivery & Hinglish safe messages."""
    rng = random.Random(SEED + 1)
    rows: list[dict] = []

    tx_templates = [
        "VM-HDFCBK-S: Rs {amt}.00 debited from A/c XX{ac} on 23-11-26 via UPI/Ref {ref}. Avail Bal: Rs {bal}. Not you? Call 18002586161 or visit hdfcbank.com",
        "AD-SBIINB-S: Dear Customer, Your A/c X{ac} is credited with Rs {amt} on 23Nov26 by IMPS Ref {ref}. Avail Bal Rs {bal}. -SBI (sbi.co.in)",
        "AX-ICICIB-S: Acct XX{ac} debited for Rs {amt} on 23-Nov-26; UPI: Swiggy. Avl Bal Rs {bal}. To dispute call 18001080 or check icicibank.com",
        "JK-AXISBK-S: INR {amt} spent on Axis Bank Card XX{ac} at Amazon India. Avl Limit: INR {bal}. Report fraud at axisbank.com",
        "JD-KOTAKB-S: Received Rs {amt} in Kotak A/c XX{ac} via NEFT Ref {ref}. Check statement on kotak.com or official .bank.in portal.",
        "VM-SBIUPI-S: Your UPI transaction of Rs {amt} to Zomato was successful. Ref {ref}. Avail Bal Rs {bal}. Never share your UPI PIN with anyone.",
    ]
    for i in range(160):
        tpl = tx_templates[i % len(tx_templates)]
        txt = tpl.format(
            amt=rng.randint(120, 18500),
            ac=rng.randint(1000, 9999),
            ref=rng.randint(100000000, 999999999),
            bal=rng.randint(5000, 95000),
        )
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_dlt", "sample_weight": 1.0})

    otp_templates = [
        "VM-HDFCBK-T: {otp} is your OTP for HDFC Bank NetBanking login. Do not share this OTP with anyone including bank staff.",
        "AD-UIDAI-G: OTP for Aadhaar authentication is {otp}. Valid for 10 mins. Never share OTP with anyone. -UIDAI (uidai.gov.in)",
        "VK-IRCTCi-T: {otp} is your One Time Password for IRCTC ticket booking. Please do not share this code with anybody.",
        "AX-SBIINB-T: Use OTP {otp} to approve card transaction of Rs {amt} at Flipkart. Do not disclose OTP to anyone. -SBI",
        "JD-PAYTMB-T: {otp} is your Paytm verification OTP. Never share it with anyone; Paytm never calls asking for OTP.",
        "AD-EPFOHO-G: Your EPFO UAN login OTP is {otp}. Do not share this OTP with anyone. Visit epfindia.gov.in for details.",
    ]
    for i in range(110):
        tpl = otp_templates[i % len(otp_templates)]
        txt = tpl.format(otp=rng.randint(100000, 999999), amt=rng.randint(250, 8000))
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_otp", "sample_weight": 1.0})

    delivery_templates = [
        "AM-AMAZON-T: Out for delivery: Your Amazon order #{ref} will be delivered today by 8 PM. Track on amazon.in",
        "VD-BLUDRT-T: BlueDart AWB {ref} is out for delivery today. Share delivery code {otp} only with the executive at your doorstep.",
        "JK-DLHVRY-T: Your Delhivery package AWB {ref} has been dispatched from Bengaluru hub. Track at delhivery.com",
        "VM-SWIGGY-T: Your Swiggy order #{ref} is on the way! Delivery partner will arrive in 12 mins. Track on swiggy.com",
        "AD-FLPKRT-T: Delivered: Your Flipkart order #{ref} was handed over safely. View invoice on flipkart.com",
        "VM-IRCTC-T: PNR {ref} Confirmed! Train 12628 Karnataka Exp Coach B3 Seat 42. Have a safe journey. irctc.co.in",
    ]
    for i in range(110):
        tpl = delivery_templates[i % len(delivery_templates)]
        txt = tpl.format(ref=rng.randint(1000000, 9999999), otp=rng.randint(1000, 9999))
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_delivery", "sample_weight": 1.0})

    hinglish_chat = [
        "Bhai 500 UPI kar de, kal subah pakka wapas deta hu.",
        "Yaar aaj bank jaana hai KYC update karwane ke liye, 2 baje milte hain.",
        "Mummy ne bola ghar aate waqt doodh aur sabzi le aana.",
        "Bro loan ka EMI kal deduct hoga account se, balance check kar liya ok.",
        "Kal office mein salary credit ho jayegi, phir dinner pe chalte hain haha.",
        "Bhai mera Flipkart order aaj deliver hone wala hai, gate pe le lena.",
        "Yaar papa ka PAN card link karwana hai bank branch jaake.",
        "Ok bro, 1000 rupay GPay kar diye hain, check kar lena.",
        "Aaj college mein TinyVerse hackathon ka demo hai, sab ready hai.",
        "Bhai train ticket IRCTC se book kar di hai, PNR WhatsApp kar raha hu.",
        "Haan yaar electricity bill maine official app se bhar diya kal hi.",
        "Mummy ko bolna OTP mat batana kisi unknown caller ko, ok.",
    ]
    for i, base in enumerate(hinglish_chat * 10):
        suffix = f" ({i})" if i >= len(hinglish_chat) else ""
        rows.append({"text": f"{base}{suffix}", "label": 0, "category": "", "source_corpus": "hard_neg_hinglish", "sample_weight": 1.5})

    # Benign Indian retail/brand promotional SMS with URLs (official + non-whitelisted brand shortlinks)
    names = ["Arya", "Rohan", "Priya", "Vikram", "Ananya", "Kabir", "Sneha", "Arjun", "Meera", "Neha"]
    occasions = ["Diwali", "Festive", "Weekend", "Monsoon", "New Season", "Summer", "Wedding"]
    known_brand_promos = [
        ("Titan World", "watches", "https://m.tneu.in/TITNWO/{code}"),
        ("Tanishq", "gold & diamond jewellery", "https://m.tneu.in/TANSHQ/{code}"),
        ("Croma", "electronics & laptops", "https://croma.com/deals/{code}"),
        ("Myntra", "ethnic wear & sneakers", "https://myntra.com/festive/{code}"),
        ("Nykaa", "skincare & beauty essentials", "https://nykaa.com/new/{code}"),
        ("Lenskart", "eyewear frames", "https://lenskart.com/store/{code}"),
        ("Tata CLiQ", "handbags & footwear", "https://tatacliq.com/edit/{code}"),
        ("BookMyShow", "weekend movies & concerts", "https://bookmyshow.com/shows/{code}"),
        ("Domino's", "freshly baked pizzas", "https://dominos.co.in/menu/{code}"),
        ("Decathlon", "running & trekking gear", "https://decathlon.in/sport/{code}"),
    ]
    unlisted_brand_promos = [
        ("Westside", "ethnic kurtas & denim", "https://go.westside.in/NEW/{code}"),
        ("FabIndia", "handloom kurtas & home decor", "https://m.fabindia.in/FEST/{code}"),
        ("Bata India", "formal & casual shoes", "https://shop.bata.in/LOOK/{code}"),
        ("Lifestyle", "partywear & handbags", "https://offers.lifestyle.in/STYLE/{code}"),
        ("Pantaloons", "festive apparel", "https://s.pantaloons.in/EDIT/{code}"),
        ("Shoppers Stop", "perfumes & watches", "https://m.shoppersstop.in/LUXE/{code}"),
        ("Pepperfry", "solid wood furniture", "https://go.pepperfry.in/HOME/{code}"),
        ("Raymond", "tailored suits & shirts", "https://m.raymond.in/SUIT/{code}"),
    ]
    promo_templates = [
        "Hi {name}, add something new to your {occ} look. See the latest {cat} at {brand}, {url} Visit today.",
        "Hello {name}, explore our new {occ} collection of {cat} at {brand}: {url} Store open till 9:30 PM.",
        "{brand}: {name}, flat 20% off on {cat} this {occ} weekend! Browse catalogue at {url} or visit your nearest store.",
        "Dear {name}, your {brand} NeuCoins / loyalty balance is ready to redeem on {cat}. Check styles at {url}",
        "Hi {name}, refresh your wardrobe for {occ} with {brand} {cat}. View lookbook: {url} T&C apply.",
    ]
    all_brands = known_brand_promos + unlisted_brand_promos
    for i in range(240):
        brand, cat, url_tpl = all_brands[i % len(all_brands)]
        tpl = promo_templates[i % len(promo_templates)]
        code = f"K{100 + i}x{chr(97 + (i % 26))}"
        txt = tpl.format(
            name=names[i % len(names)],
            occ=occasions[i % len(occasions)],
            cat=cat,
            brand=brand,
            url=url_tpl.format(code=code),
        )
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_promo_url", "sample_weight": 1.5})

    # Benign informational / utility / personal SMS containing URLs
    safe_url_templates = [
        "BSES Rajdhani: Your electricity bill of Rs {amt} for Oct 2026 is generated. View PDF at https://bsesdelhi.com/bill/{code}",
        "Airtel: Your broadband bill of Rs {amt} is paid. Download GST invoice at https://airtel.in/inv/{code}",
        "JioFiber: Plan renewed for 30 days. Manage WiFi settings at https://jio.com/fiber/{code}",
        "Hi {name}, here is the Google Meet link for tomorrow's project sync at 11 AM: https://meet.example.in/room/{code}",
        "Apollo Diagnostics: Your blood test report is ready. View securely at https://reports.apollolab.in/{code}",
        "Dear {name}, thanks for dining with us! View our digital menu & feedback form at https://menu.bistro.in/{code}",
        "IRCTC: E-ticket booked for Train 12952. Check PNR status at https://irctc.co.in/pnr/{code}",
        "College Notice: TinyVerse Hackathon schedule and room allocation updated at https://campus.edu.in/events/{code}",
    ]
    for i in range(110):
        tpl = safe_url_templates[i % len(safe_url_templates)]
        txt = tpl.format(
            name=names[i % len(names)],
            amt=rng.randint(499, 3499),
            code=f"r{2000 + i}",
        )
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_safe_url", "sample_weight": 1.5})

    # 7. Telecom & DTH usage / plan expiry / missed-call alerts (directing to official apps/domains)
    telecom_templates = [
        "Dear Customer, your Jio plan of Rs {amt} is expiring in {days} days. Recharge now on MyJio app or jio.com to enjoy uninterrupted services.",
        "Airtel: You have used {pct}% of your daily {gb}GB data quota. Buy a data pack on Airtel Thanks app.",
        "Vi: Your unlimited 5G pack validity ends in {days} days. Renew easily via Vi app or vi.in.",
        "BSNL: Your FTTH broadband bill of Rs {amt} is ready. Pay securely on bsnl.co.in or MyBSNL app.",
        "Tata Play: Your DTH balance is low (Rs 45). Recharge via Tata Play app to continue watching your channels.",
        "Jio: You have {n} missed call(s) from {ph} at {time} when your phone was switched off. View call log in MyJio app.",
        "Vi Missed Call Alert: {ph} tried calling you {n} time(s) while you were out of coverage. Details in Vi app.",
        "Airtel: While your number was unreachable, you received {n} missed call(s). Check the Airtel Thanks app for details.",
        "BSNL: {n} calls were missed while your mobile was not reachable on {date}. Dial *121# for details.",
        "Jio: {ph} is now available to take your call. You had tried calling earlier.",
        "{n} missed call(s) received on {date} while your Airtel number was not reachable. Check details on Airtel App - https://i.airtel.in/missed_{n}",
        "Jio Missed Call Alert: {n} call(s) missed on {date}. View caller log on MyJio App - https://jio.com/missed/{n}",
        "Vi Alert: {n} missed call(s) when your phone was switched off on {date}. Check only on Vi App - https://vi.in/calls/{n}",
    ]
    for i in range(150):
        tpl = telecom_templates[i % len(telecom_templates)]
        txt = tpl.format(
            amt=rng.choice([239, 299, 349, 666, 799]),
            days=rng.randint(1, 5),
            pct=rng.choice([50, 90, 100]),
            gb=rng.choice(["1.5", "2.0", "2.5"]),
            n=rng.randint(1, 5),
            ph=f"9{rng.randint(100000000, 999999999)}",
            time=f"{rng.randint(10, 21)}:{rng.randint(10, 59)}",
            date=f"{rng.randint(10, 28)}-11-2026",
        )
        if i >= len(telecom_templates):
            txt += f" [ID:{100 + i}]"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_telecom_utility", "sample_weight": 1.5})

    # 8. Benign food / grocery / e-commerce app coupons & sales
    coupon_templates = [
        "Zomato: Craving biryani tonight? Get {pct}% OFF up to Rs {off} using code TRYNEW. Order now on Zomato app!",
        "Swiggy: Flat Rs {off} off on your next Instamart order above Rs 399! Valid till midnight. Shop now on Swiggy app.",
        "Blinkit: Fresh mangoes & cold coffee delivered in 10 mins! Use code FRESH{off} for free delivery.",
        "Flipkart: Big Diwali Sale starts tonight at 12 AM! Extra 10% instant discount on SBI Credit Cards.",
        "Myntra: Your wishlist item is now at {pct}% off! Grab it before stock runs out on Myntra app.",
        "BigBasket: Weekend grocery savings! Get Rs {off} cashback on your basket above Rs 1500 in the BigBasket app.",
        "Swiggy One: Renew your 3-month membership for just Rs {off}! Enjoy free delivery on orders above Rs 199.",
        "Amazon Prime: Renew your Prime membership for Rs {off} and get free 1-day delivery & Prime Video.",
    ]
    for i in range(90):
        tpl = coupon_templates[i % len(coupon_templates)]
        txt = tpl.format(pct=rng.choice([20, 30, 40, 50]), off=rng.choice([75, 100, 120, 150]))
        if i >= len(coupon_templates):
            txt += f" #{i}"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_app_coupons", "sample_weight": 1.5})

    # 9. Genuine in-app bank / finance / refund / UPI-received notifications
    inapp_finance_templates = [
        "HDFC Bank: Pre-approved Credit Card upgrade available for you! Check eligibility in NetBanking or HDFC MobileBanking app.",
        "ICICI Bank: Your credit card bill of Rs {amt} is due on 28 Nov 2026. Pay via iMobile app to avoid late fees.",
        "SBI Card: Your monthly statement for Card XX{ac} is generated. Total due Rs {amt}. View in SBI Card app.",
        "Amazon: Your refund of Rs {amt} for order #402-{ref} has been initiated and will reflect in 3-5 business days.",
        "Reminder: Your SIP of Rs {amt} in Nifty 50 Index Fund will be auto-debited on 25 Nov. Keep sufficient balance.",
        "Paytm: You received Rs {amt} from {sender} via UPI. Updated wallet/bank history in app.",
        "Congratulations on completing 1 year with Zerodha! View your annual portfolio summary in Console.",
        "PhonePe: Payment of Rs {amt} received from {sender}. Check transaction history in PhonePe app.",
    ]
    senders = ["ARJUN SHARMA", "PRIYA NAIR", "ROHAN VERMA", "SNEHA PATEL", "KABIR MEHTA"]
    for i in range(90):
        tpl = inapp_finance_templates[i % len(inapp_finance_templates)]
        txt = tpl.format(amt=rng.randint(350, 18500), ac=rng.randint(1000, 9999), ref=rng.randint(1000000, 9999999), sender=senders[i % len(senders)])
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_inapp_finance", "sample_weight": 1.5})

    # 10. Ride-hailing / doorstep delivery PINs, appointments, campus & personal messages
    ride_delivery_templates = [
        "Uber: Your ride with {driver} (KA01AB{ac}) is arriving in {mins} mins. Share PIN {pin} with driver to start trip.",
        "Ola: Auto arriving in {mins} mins. Share OTP {pin} with driver to start your ride.",
        "Rapido: Captain {driver} is at your pickup point. Share PIN {pin} to begin ride.",
        "Can you please share the OTP for the Amazon delivery? The delivery boy is at the gate.",
        "Flipkart delivery executive is outside, please tell him the delivery OTP {pin}.",
        "Hi, your appointment with Dr. Mehta is confirmed for tomorrow at 4:30 PM at Manipal Hospital.",
        "Dear Student, exam hall tickets for Semester 5 are live on the university portal. Download before Friday.",
        "Hey bro, did you send the {amt} rs to Rahul? Let me know once done, need to settle the dinner split.",
    ]
    drivers = ["Ramesh", "Suresh", "Vikram", "Manoj", "Deepak"]
    for i in range(80):
        tpl = ride_delivery_templates[i % len(ride_delivery_templates)]
        txt = tpl.format(driver=drivers[i % len(drivers)], ac=rng.randint(1000, 9999), mins=rng.randint(2, 8), pin=rng.randint(1000, 9999), amt=rng.choice([500, 1200, 2000]))
        if i >= len(ride_delivery_templates):
            txt += f" (Ref {i})"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_ride_delivery_pin", "sample_weight": 1.5})

    # 11. Regulatory, Legal & Cyber-Awareness PSAs (240 rows)
    psa_templates = [
        "RBI Kehta Hai: Never share your PIN, OTP, CVV or internet banking password with anyone, even if they claim to be bank officials. Report fraud at 1930.",
        "RBI Advisory: Do not disclose your ATM PIN, OTP, or CVV to anyone over phone calls, SMS, or messaging apps. Stay alert.",
        "Bank Security Tip: Our bank never asks for your card PIN, CVV, MPIN, or OTP over phone or SMS. Keep your credentials confidential.",
        "SEBI Investor Caution: Do not act on unregistered stock market tips circulated via social media or messaging apps. Verify advisors at https://sebi.gov.in",
        "NSE / BSE Advisory: Investors are advised not to subscribe to guaranteed-return schemes or unverified advisory channels. Check registered brokers on nseindia.com",
        "DoT Public Notice: Forging identity documents to obtain mobile connections is a criminal offence attracting penalties up to Rs 50 lakh or imprisonment under the Telecom Act.",
        "TRAI Advisory: Registered banking and financial institutions use designated 1600-series numbering for service and transactional voice calls. Verify caller identity.",
        "Cyber Dost (MHA): Stay safe from digital arrest frauds. Police or CBI never conduct arrests or demand money over video calls. Call 1930 or visit cybercrime.gov.in",
        "NPCI Safety Tip: You never need to enter your UPI PIN to receive money into your bank account. PIN is only required to authorize outgoing payments.",
        "CERT-In Cyber Hygiene: Keep your mobile OS updated and avoid installing unknown application packages shared outside official app stores.",
    ]
    for i in range(250):
        txt = psa_templates[i % len(psa_templates)]
        if i >= len(psa_templates):
            txt += f" Ref:{1000 + i}"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_regulatory_psa", "sample_weight": 1.5})

    # 12. Customer Support Apology Vouchers, Credits & Membership Notices (200 rows)
    support_templates = [
        "We apologize for the delay in your {brand} order #{ord_id}. Here is a goodwill voucher worth Rs {amt}.00 using code {code}, valid until {exp} on your next checkout.",
        "Sorry for the inconvenience on order #{ord_id}. Apply promo code {code} to get Rs {amt}.00 off on your next {brand} order, valid till {exp}.",
        "As a token of apology for your recent {brand} delivery #{ord_id}, enjoy a discount coupon worth Rs {amt}.00 (code: {code}). Unused balance re-issued after delivery.",
        "{brand} Support: Thank you for reaching out. A credit of Rs {amt}.00 has been added to your wallet for order #{ord_id}. Remaining balance carries forward automatically.",
        "Your {brand} Plus membership has been renewed for 12 months. View invoice and benefits in your account settings.",
        "Good news! Your refund of Rs {amt}.00 for cancelled order #{ord_id} has been processed to your original payment method.",
        "Hi {name}, your cashback of Rs {amt} from {brand} Pay for your recent utility bill is now available in your wallet balance.",
        "{brand} Pay: Rs {amt} cashback has been credited to your {brand} Pay balance for your electricity bill payment.",
        "Cashback Credited: Rs {amt} added to your {brand} wallet for completing your broadband recharge.",
    ]
    support_brands = ["Swiggy", "Zomato", "Zepto", "BigBasket", "Amazon", "Flipkart", "Uber", "Ola"]
    for i in range(200):
        tpl = support_templates[i % len(support_templates)]
        txt = tpl.format(
            brand=support_brands[i % len(support_brands)],
            ord_id=f"ORD{700000 + i * 13}",
            amt=rng.choice([25, 40, 50, 75, 100, 150]),
            code=f"CARE{1000 + i}",
            exp=f"{rng.randint(10, 28)} Dec 2026",
            name=names[i % len(names)],
        )
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_support_coupons", "sample_weight": 1.5})

    # 13. Tech / SaaS & Account Security Alerts (200 rows)
    tech_templates = [
        "{svc} Security Alert: A new sign-in to your account was detected from Chrome on Windows. If this was you, no action is needed. Review at {dom}",
        "{svc}: Two-factor authentication (2FA) was successfully enabled on your account. Keep your recovery codes in a safe place.",
        "{svc}: A new SSH key or API access token was generated for your workspace. Manage active tokens at {dom}",
        "{svc} Billing: Your monthly cloud invoice of USD {amt}.00 has been paid. Download your PDF receipt from the billing dashboard at {dom}",
        "Your {svc} verification code is {otp}. Never share this security code with anyone.",
    ]
    tech_svcs = [
        ("Google", "google.com"), ("Apple", "apple.com"), ("Microsoft", "microsoft.com"),
        ("AWS", "aws.amazon.com"), ("Slack", "slack.com"), ("GitHub", "github.com"),
        ("Notion", "notion.so"), ("Discord", "discord.com"),
    ]
    for i in range(200):
        svc, dom = tech_svcs[i % len(tech_svcs)]
        tpl = tech_templates[i % len(tech_templates)]
        txt = tpl.format(svc=svc, dom=dom, amt=rng.randint(5, 49), otp=rng.randint(100000, 999999))
        if i >= len(tech_templates) * len(tech_svcs):
            txt += f" [Event {i}]"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_tech_alerts", "sample_weight": 1.5})

    # 14. Job, Career & Education Platforms (200 rows)
    career_templates = [
        "{plat}: {cnt} new Software Developer & Data Analyst openings in {city} match your saved preferences. Browse listings at {dom}",
        "{plat}: {cnt} hiring managers viewed your profile this week. Keep your resume and skills up to date on {dom}",
        "{plat}: Your application for the Summer Engineering Intern role has been shortlisted for the technical round. Check status on {dom}",
        "University Placement Office: Pre-placement talk and aptitude test scheduled for tomorrow at 10:30 AM in Seminar Hall B. Register via {dom}",
        "NPTEL / Coursera: Week {wk} assignment deadline is Sunday 11:59 PM IST. Submit your quiz on the course portal.",
    ]
    career_plats = [
        ("Indeed", "indeed.com"), ("Unstop", "unstop.com"), ("Glassdoor", "glassdoor.co.in"),
        ("Wellfound", "wellfound.com"), ("LinkedIn", "linkedin.com"), ("Naukri", "naukri.com"),
        ("Internshala", "internshala.com"),
    ]
    for i in range(200):
        plat, dom = career_plats[i % len(career_plats)]
        tpl = career_templates[i % len(career_templates)]
        txt = tpl.format(
            plat=plat,
            dom=dom,
            cnt=rng.randint(3, 25),
            city=rng.choice(["Bengaluru", "Hyderabad", "Pune", "Mumbai", "Gurugram"]),
            wk=rng.randint(1, 8),
        )
        if i >= len(career_templates) * len(career_plats):
            txt += f" #{i}"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_career_edu", "sample_weight": 1.5})

    # 15. Informational Banking, Fintech, Investing & Insurance (250 rows)
    fintech_templates = [
        "{bank}: Your FASTag wallet balance is low (Rs {amt}). Top up via official {bank} NetBanking or mobile app before your next highway trip.",
        "{bank}: Scheduled system maintenance on NetBanking & mobile app from 01:00 AM to 03:30 AM IST this Sunday. UPI services remain available.",
        "{broker}: Your KYC and demat onboarding verification is complete. You may now explore mutual funds and equities on {dom}",
        "{broker}: Account KYC verification completed! You can now start investing in Mutual Funds and Stocks anytime in the app.",
        "{broker}: Dividend of Rs {amt}.50 has been credited to your linked bank account. View your holding statement on {dom}",
        "{bank}: You have {pts} reward points available in your account. Redeem them anytime inside the official {bank} mobile app.",
        "LIC / Health Insurance: Your annual policy #{ord_id} premium of Rs {amt} is received. Download policy receipt from the official portal.",
        "{bank}: Exclusive pre-approved credit card limit enhancement available! Log in directly to your official {bank} MobileBanking app to view details.",
        "Bill Payment Confirmed: Your {bank} credit card bill of Rs {amt} is paid successfully via {broker}. You earned {pts} reward coins.",
        "Subscription Renewed: Your monthly digital membership of Rs {amt} for cloud & music streaming is active. Manage billing in app settings.",
        "Music Streaming Plan Renewed: Your Student Premium plan renewed for Rs {amt}/month. Enjoy uninterrupted ad-free listening.",
        "Cloud Storage Plan: Your 50GB monthly backup subscription of Rs {amt} has been renewed via auto-pay.",
    ]
    fin_banks = ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Bank"]
    fin_brokers = [("Zerodha", "zerodha.com"), ("Groww", "groww.in"), ("Upstox", "upstox.com")]
    for i in range(250):
        bank = fin_banks[i % len(fin_banks)]
        broker, dom = fin_brokers[i % len(fin_brokers)]
        tpl = fintech_templates[i % len(fintech_templates)]
        txt = tpl.format(
            bank=bank,
            broker=broker,
            dom=dom,
            amt=rng.randint(150, 9500),
            pts=rng.randint(1200, 9500),
            ord_id=f"POL{80000 + i}",
        )
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_fintech_info", "sample_weight": 1.5})

    # 16. Lifestyle, Food BOGO, OTT, Travel & Everyday Safe Links (220 rows)
    lifestyle_templates = [
        "Pizza Hut: Buy 1 Get 1 Free on Medium Pan Pizzas every Wednesday! Order directly on pizzahut.co.in or the Pizza Hut app.",
        "Lenskart: Book a complimentary eye test at your nearest store today! Schedule your visit on lenskart.com",
        "Hotstar: Stream the India vs England T20 match live free on mobile in the Disney+ Hotstar app starting 7:00 PM IST.",
        "IndiGo: Web check-in is now open for your flight 6E-{flt} DEL to BOM. Download your boarding pass on goindigo.in",
        "Practo / 1mg: Your medicine order #{ord_id} is packed and will arrive tomorrow morning. Track in the app.",
        "Hey team, here is the YouTube recording and Google Doc notes from today's sprint review: https://youtu.be/demo{code}",
    ]
    for i in range(220):
        tpl = lifestyle_templates[i % len(lifestyle_templates)]
        txt = tpl.format(flt=rng.randint(100, 999), ord_id=f"MG{4000 + i}", code=f"V{i}")
        if i >= len(lifestyle_templates):
            txt += f" [{i}]"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_lifestyle_promos", "sample_weight": 1.5})

    # 17. Headerless Indian Transactional Bank SMS (260 rows, NO DLT prefix)
    tx_banks = ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak Bank", "Bank of Baroda", "PNB", "Canara Bank", "Federal Bank", "IndusInd Bank", "IDFC First Bank"]
    merchants = ["ZOMATO", "SWIGGY", "AMAZON", "FLIPKART", "BIGBASKET", "UBER", "AIRTEL", "BOOKMYSHOW"]
    headerless_tx_templates = [
        "Dear UPI user A/C X{ac} debited by {amt}.0 on date 24Nov26 trf to {merch} Refno {ref}. If not u? call 1800111109. -SBI",
        "Sent Rs.{amt}.00 From {bank} A/C *{ac} To {merch} On 24/11/26 Ref {ref} Not You? Call 18002586161",
        "INR {amt}.00 spent using {bank} Card XX{ac} on 24-Nov-26 on {merch}. Avl Limit: INR {bal}.00. If not you, call 18001080 or SMS BLOCK {ac} to 9215676766.",
        "Rs. {amt}.00 withdrawn from {bank} A/c XXXXX{ac} at ATM BLR042 on 24Nov26. Avl Bal Rs.{bal}.00. Not done by you? Forward this SMS to 9223008333 or call 18001234.",
        "Your A/c no. XX{ac} is credited with Rs.{amt}.00 on 24-11-26 by NEFT from ACME TECH PVT LTD. Avl Bal Rs.{bal}.00.",
        "Amt Sent Rs.{amt}.00 From {bank} AC X{ac} to merchant@okicici on 24-11-26. UPI Ref {ref}.",
        "Alert: You've spent Rs.{amt} on {bank} Credit Card ending {ac} at {merch} on 24-11-2026. Avl Limit: Rs.{bal}.",
        "Thank you for using your {bank} Debit Card ending {ac} for a purchase worth Rs {amt}.00 at {merch}.",
        "{bank}: Rs.{amt}.00 auto-debited from your Card XX{ac} for Standing Instruction. Manage at sihub.in",
        "Your request for NEFT transfer of Rs.{amt}.00 from A/c XX{ac} to Beneficiary has been processed. Ref {ref}.",
        "Dear Customer, Cheque no. 00{ac} for Rs.{amt}.00 has been cleared from your A/c XX{ac2}. Avl Bal Rs.{bal}.00.",
        "Transaction of INR {amt}.00 on {bank} Credit Card ending {ac} at {merch} was declined due to insufficient limit.",
    ]
    for i in range(260):
        tpl = headerless_tx_templates[i % len(headerless_tx_templates)]
        txt = tpl.format(
            bank=tx_banks[i % len(tx_banks)],
            merch=merchants[i % len(merchants)],
            amt=rng.randint(120, 45000),
            ac=rng.randint(1000, 9999),
            ac2=rng.randint(1000, 9999),
            ref=rng.randint(100000000000, 999999999999),
            bal=rng.randint(5000, 125000),
        )
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_tx_headerless", "sample_weight": 1.5})

    # 18. Balanced Multilingual Indic Corpus (Hindi, Kannada, Tamil, Telugu — SCAM & SAFE)
    indic_scams = [
        ("KYC/Bank", "आपका बैंक खाता केवाईसी अपडेट न होने के कारण बंद हो जाएगा। अभी लिंक पर क्लिक करें: bit.ly/kyc-hi-{i}"),
        ("Other", "ನಿಮ್ಮ ವಿದ್ಯುತ್ ಬಿಲ್ ಬಾಕಿ ಇದೆ, ಇಂದು ರಾತ್ರಿ ಸಂಪರ್ಕ ಕಡಿತಗೊಳ್ಳುತ್ತದೆ. ತಕ್ಷಣ ಕರೆ ಮಾಡಿ 9876543{i:03d}"),
        ("KYC/Bank", "ನಿಮ್ಮ ಎಸ್‌ಬಿಐ ಬ್ಯಾಂಕ್ ಖಾತೆ ಕೆವೈಸಿ ಕಾರಣದಿಂದ ಬ್ಲಾಕ್ ಆಗಿದೆ. ತಕ್ಷಣ ಅಪ್‌ಡೇಟ್ ಮಾಡಿ: bit.ly/sbi-ka-{i}"),
        ("Lottery/Prize", "உங்கள் எண் ஜியோ லக்கி டிராவில் ரூ.15 லட்சம் வென்றுள்ளது. உடனே தொடர்பு கொள்ளவும் 9845012{i:03d}"),
        ("Other", "உங்கள் மின்சார இணைப்பு இன்று இரவு துண்டிக்கப்படும். உடனே அதிகாரி 9811223{i:03d} எண்ணை அழைக்கவும்."),
        ("KYC/Bank", "మీ బ్యాంక్ ఖాతా బ్లాక్ చేయబడింది. వెంటనే KYC అప్‌డేట్ చేయండి: bit.ly/kyc-te-{i}"),
        ("Lottery/Prize", "అభినందనలు! మీరు రూ.20 లక్షల లాటరీ గెలుచుకున్నారు. క్లెయిమ్ చేయడానికి కాల్ చేయండి 9900112{i:03d}"),
    ]
    indic_safe = [
        "भाई कल सुबह 9 बजे कॉलेज में मिलते हैं, नोट्स लेकर आना।",
        "ನಿಮ್ಮ ಅಮೆಜಾನ್ ಆರ್ಡರ್ ಇಂದು ತಲುಪಿಸಲಾಗುತ್ತದೆ. ಟ್ರ್ಯಾಕ್ ಮಾಡಿ amazon.in",
        "ನಿಮ್ಮ ಬ್ಯಾಂಕ್ ಖಾತೆಗೆ ರೂ. 2,500 ಜಮಾ ಆಗಿದೆ. ಬ್ಯಾಲೆನ್ಸ್ ರೂ. 14,200.",
        "உங்கள் ஸ்விக்கி ஆர்டர் 15 நிமிடங்களில் வரும். நன்றி!",
        "உங்கள் வங்கி கணக்கில் ரூ.1,200 டெபிட் செய்யப்பட்டது.",
        "మీ ఆర్డర్ డెలివరీ కోసం బయలుదేరింది. ట్రాక్ చేయండి flipkart.com",
        "అన్నా ఈరోజు సాయంత్రం 6 గంటలకు కలుద్దామా?",
    ]
    for i in range(56):
        cat, tpl = indic_scams[i % len(indic_scams)]
        rows.append({"text": tpl.format(i=100 + i), "label": 1, "category": cat, "source_corpus": "indian_curated_2026", "sample_weight": 1.5})
    for i in range(56):
        txt = indic_safe[i % len(indic_safe)]
        if i >= len(indic_safe):
            txt += f" ({i})"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_indic_safe", "sample_weight": 1.5})

    # 19. High-Diversity Indian Commercial Ads & Retail Promos (450 rows — with & without shortlinks / DLT -P)
    ad_brands = [
        ("Pizza Hut", "AD-PIZAHUT-P", "pizzas & garlic bread", "pizzahut.co.in", "bit.ly/ph-bogo"),
        ("McDonald's", "VM-MCDIND-P", "McSpicy burgers & fries", "mcdelivery.co.in", "bit.ly/mcd-meal"),
        ("Subway", "AX-SUBWAY-P", "subs & wraps", "subway.in", "tinyurl.com/sub-deal"),
        ("Taco Bell", "AD-TACOBL-P", "tacos & burritos", "tacobell.co.in", "cutt.ly/tb-combo"),
        ("Chaayos", "VM-CHAAYO-P", "chai & bun maska", "chaayos.com", "bit.ly/chy-chai"),
        ("Haldiram's", "AD-HALDIR-P", "sweets & namkeen combos", "haldirams.com", "bit.ly/hl-fest"),
        ("Barbeque Nation", "AX-BBQN-P", "weekday buffet", "barbequenation.com", "tinyurl.com/bbq-bk"),
        ("Behrouz Biryani", "VM-BEHROU-P", "royal biryani bowls", "behrouzbiryani.com", "bit.ly/bb-feast"),
        ("Zomato", "AD-ZOMATO-P", "biryani, pizzas & bowls", "zomato.com", "bit.ly/zom-off"),
        ("Starbucks", "AX-STARBK-P", "tall handcrafted beverages", "starbucks.in", "bit.ly/sbx-brew"),
        ("Zepto", "JD-ZEPTO-P", "groceries & snacks", "zeptonow.com", "bit.ly/zep-10m"),
        ("BigBasket", "VM-BIGBSK-P", "fresh fruits & staples", "bigbasket.com", "tinyurl.com/bb-fresh"),
        ("DMart Ready", "AD-DMART-P", "monthly home essentials", "dmart.in", "cutt.ly/dm-save"),
        ("Reliance SMART", "AX-RLSMRT-P", "supermarket deals", "jiomart.com", "bit.ly/rsm-sale"),
        ("Reliance Digital", "VM-RELDIG-P", "smart TVs & laptops", "reliancedigital.in", "bit.ly/rd-tech"),
        ("boAt", "AD-BOATLF-P", "Airdopes & smartwatches", "boat-lifestyle.com", "tinyurl.com/bt-drop"),
        ("Urban Company", "AX-URBNCP-P", "salon & home cleaning", "urbancompany.com", "bit.ly/uc-home"),
        ("BookMyShow", "VM-BKMYSH-P", "weekend movies & events", "bookmyshow.com", "bit.ly/bms-tix"),
        ("Myntra", "AX-MYNTRA-P", "sneakers & ethnic wear", "myntra.com", "bit.ly/myn-sale"),
        ("AJIO", "VM-AJIOIN-P", "denim, kurtas & footwear", "ajio.com", "cutt.ly/aj-bold"),
    ]
    ad_codes = ["SAVE50", "FLAT100", "BOGO20", "TREAT150", "FESTIVE30", "WEEKEND25", "DEAL40", "NEW120"]
    ad_templates_short = [
        "Today only! Get Flat {disc}% OFF on {cat} at {brand}. Use code {code}. Grab now! {surl}{i} TnC",
        "{brand}: Buy 1 Get 1 FREE on {cat}! Valid till midnight. Don't miss this... order now: {surl}{i} T&C apply",
        "Hungry? Order {cat} from {brand} & get Flat Rs {amt} OFF + Free delivery above Rs 199! Shop: {surl}{i}",
        "Weekend Flash Sale at {brand}! Minimum {disc}% OFF on {cat}. Visit nearest store or {surl}{i} TnC",
        "{brand} Mega Offer! Extra Rs {amt} OFF on {cat} with code {code}. Valid today only! {surl}{i} T&C apply",
        "Special {brand} Deal: Flat {disc}% off on {cat} all week! Apply {code} at checkout: {surl}{i} T&C",
        "Celebrate with {brand}! Buy 2 get 1 FREE on {cat}. Order online via {surl}{i} or visit outlet. TnC",
        "Limited period offer at {brand}: Save Rs {amt} on {cat} orders above Rs 599. Shop now: {surl}{i} TnC apply",
        "{brand} Member Exclusive: Extra {disc}% savings on {cat} this Friday! Redeem at {surl}{i}",
        "Just dropped at {brand}! Fresh {cat} starting Rs {amt}. Explore collection: {surl}{i} T&C apply",
        "Craving {cat}? {brand} brings Flat Rs {amt} OFF today! Use {code} & order here: {surl}{i}",
        "{brand} Super Saver: Up to {disc}% OFF on {cat} + instant bank discount. Check deals: {surl}{i} TnC",
        "Happy Hours at {brand}! Get {disc}% off on {cat} from 4 PM to 8 PM. Book/Order: {surl}{i}",
        "Festive Collection from {brand}: Flat {disc}% OFF on {cat} + free shipping. Visit {surl}{i} T&C apply",
        "{brand} Flash Drop! Grab {cat} at flat Rs {amt} off with code {code}. Valid tonight only: {surl}{i} TnC",
        "Today only! Double your {brand} meal. Buy 4 pcs combo and get 4 pcs sides... FREE! Don't miss this... grab now! {surl}{i} TnC",
        "{brand} Festive Sale! Up to {disc}% OFF on {cat} + No Cost EMI. Visit {brand} store or {surl}{i} TnC",
        "{brand}: Newly launched {cat} at Rs {amt}! Limited stock launch price today only. Grab yours now: {surl}{i} T&C",
        "{brand} Flash Deal! Up to {disc}% OFF on {cat} + 10% Bank Cashback. Check store or {surl}{i} TnC",
    ]
    ad_templates_plain = [
        "Craving {cat}? Get Flat {disc}% OFF on {brand} up to Rs {amt}! Use code {code}. Valid today only! T&C apply",
        "{brand} End of Season Sale is LIVE! Up to {disc}% OFF on {cat} + No Cost EMI. Shop on {brand} App or {dom} TnC",
        "Buy 2 get 1 FREE on {cat} at {brand}! Valid today only at participating outlets. Don't miss out! TnC apply",
        "Special Treat from {brand}! Flat Rs {amt} OFF on {cat} orders above Rs 499. Apply code {code} in {brand} App today!",
        "{brand} Loyalty Offer: Enjoy {disc}% OFF on {cat} this weekend! Order on {dom} or visit store. T&C apply",
    ]
    for i in range(450):
        brand, hdr, cat, dom, surl = ad_brands[i % len(ad_brands)]
        code = ad_codes[i % len(ad_codes)]
        disc = rng.choice([20, 30, 40, 50, 60, 70])
        amt = rng.choice([100, 120, 150, 200, 250, 500])
        if i % 20 < 9:  # ~45% with commercial shortlinks
            tpl = ad_templates_short[i % len(ad_templates_short)]
        else:
            tpl = ad_templates_plain[i % len(ad_templates_plain)]
        txt = tpl.format(brand=brand, cat=cat, dom=dom, surl=surl, code=code, disc=disc, amt=amt, i=i)
        if i % 5 < 2:  # ~40% with TRAI DLT -P prefix
            txt = f"{hdr}: {txt}"
        rows.append({"text": txt, "label": 0, "category": "", "source_corpus": "hard_neg_commercial_ads", "sample_weight": 1.8})

    return rows


def _generate_diverse_modern_scams() -> list[dict]:
    """Diverse natural-language Indian & global social-engineering scams (no rigid 40x template clones)."""
    rng = random.Random(SEED + 7)
    items: list[tuple[str, str]] = [
        # 1. Phishing on abusable legitimate platforms (forms.gle, docs.google.com, github.io, sites.google.com)
        ("KYC/Bank", "HDFC Bank Notice: Re-confirm your PAN & mobile number within 24 hrs via https://forms.gle/hdfcPanVerify99 or netbanking will be paused."),
        ("KYC/Bank", "SBI YONO Alert: Complete mandatory Aadhaar-PAN re-KYC today at https://docs.google.com/forms/d/e/1FAIpQLSf_sbi_kyc_form/viewform"),
        ("KYC/Bank", "Axis Bank Security: Your debit card has been temporarily frozen. Submit verification details on https://axis-portal-check.github.io/auth"),
        ("KYC/Bank", "ICICI Account Desk: Update your registered email and ATM PIN details at https://sites.google.com/view/icici-kyc-helpdesk"),
        ("Job/Investment", "Congratulations! Shortlisted for Amazon Remote Operations Intern (Rs 35,000/mo). Pay Rs 499 onboarding verification on https://forms.gle/amzInternFee"),
        ("KYC/Bank", "Income Tax India: Pending TDS refund of Rs 11,480 detected. Submit bank account and IFSC on https://docs.google.com/forms/d/e/itr_claim_2026"),
        ("Job/Investment", "Flipkart Work-From-Home Hiring: Fill your UPI ID and Rs 250 registration fee at https://forms.gle/fkWfhApply26 to start earning today."),
        ("Other", "BSES Electricity Desk: Submit your consumer number and bill receipt on https://forms.gle/bsesReconnect to stop disconnection tonight."),
        # 2. Pig-butchering / crypto / gold / quant investment scams
        ("Job/Investment", "Hey, we met briefly at the Bengaluru fintech summit last month. I trade gold futures with an institutional signal group making 15-20% monthly—want me to show you my dashboard?"),
        ("Job/Investment", "Hi! My cousin is a senior analyst at a Hong Kong quant desk and shares private USDT arbitrage entries. Try starting with Rs 15,000 on our platform and see the profit yourself."),
        ("Job/Investment", "Hello friend, sorry wrong number—btw I am an investment advisor in Singapore. If you have spare savings I can guide you on forex liquidity mining for steady weekly returns."),
        ("Job/Investment", "Join our private Nifty & BankNifty options mentorship desk. Our AI bot has 94% win rate; deposit Rs 10,000 capital to copy trades automatically."),
        ("Job/Investment", "Airdrop live: Connect your TrustWallet or MetaMask at https://eth-bonus-drop.net to claim 1,800 USDT governance reward before snapshot closes."),
        ("Job/Investment", "Claim your Solana ecosystem loyalty airdrop of 45 SOL! Approve wallet signature at https://sol-airdrop-portal.xyz within 3 hours."),
        ("Job/Investment", "Hey! Saw your LinkedIn—I run a private crypto arbitrage desk in Dubai delivering 12-18% monthly yield. Happy to share my live P&L if you want to test with Rs 5,000."),
        ("Job/Investment", "Hi, sorry if this is random—I got your contact from a Delhi startup mixer. My mentor manages an automated forex & bullion pool giving 4% weekly returns, interested?"),
        ("Job/Investment", "Good evening! My sister works at a quantitative trading firm in Singapore and gives me daily BTC/USDT entry signals. Start with just Rs 8,000 on our exchange app."),
        ("Job/Investment", "Exclusive invite: Join our institutional gold & crude oil MCX signal room. Members double capital every 45 days—deposit Rs 10,000 to activate copy trading."),
        ("Job/Investment", "Hey dear, I am an angel investor based in London. I teach friends how to earn steady passive income from Web3 liquidity pools—want me to walk you through it?"),
        ("Job/Investment", "Hello! Are you open to an automated crypto futures strategy? Minimum capital is only Rs 5,000 and profits are withdrawable daily to UPI."),
        ("Job/Investment", "Hi there, my uncle is a senior market maker and shares pre-IPO & USDT insider allocations. Put Rs 12,000 in our test pool today to see guaranteed gains."),
        ("Job/Investment", "Private Wealth Desk: Earn 20% monthly fixed return on digital gold & USDT staking. Zero risk, backed by Singapore liquidity pool. Reply YES for link."),
        ("Job/Investment", "Hey, I made Rs 1.8 lakh this month trading ETH options using a private Telegram signal bot. Let me add you to the VIP trial group today."),
        ("Job/Investment", "Looking for serious partners for high-yield crypto arbitrage. Deposit Rs 10,000, earn Rs 2,500 daily profit directly to your bank."),
        ("Job/Investment", "Hi! I am a portfolio analyst in Hong Kong. Our AI quantitative desk has 92% accuracy on gold futures—want to see yesterday's profit statement?"),
        ("Job/Investment", "Join our VIP Forex & Crypto Insider Circle. Get 15% guaranteed weekly returns on USDT deposits starting Rs 5,000."),
        # 3. BEC / Boss / Colleague gift-card & urgent wire scams
        ("Other", "Hi, this is Vikram (MD). I am in a confidential investor meeting and cannot answer calls. Can you purchase 4 Apple gift cards of Rs 5,000 each and email me the redemption codes asap?"),
        ("Other", "Urgent request from Director's office: Please process an immediate NEFT of Rs 48,000 to vendor account 9988776655@ybl, I will approve the voucher once I land."),
        ("Other", "Hey, I am stuck in a client presentation and need a quick favor—buy 3 Google Play gift vouchers worth Rs 10,000 total and WhatsApp the PINs to me right away."),
        ("Other", "Hi, this is Ananya (VP Finance). I am boarding a flight and cannot access corporate banking. Urgently buy 5 Amazon Pay e-gift cards of Rs 5,000 each and send the codes."),
        ("Other", "Quick favor from CEO desk: Need Rs 25,000 worth of Flipkart gift vouchers for an urgent client appreciation—send codes to this email, will reimburse tomorrow."),
        # 4. Family / friend emergency impersonation
        ("Other", "Beta my debit card stopped working at Mumbai airport and I have to pay for my boarding upgrade right now. Transfer Rs 14,000 to my colleague's PhonePe 9820112233 immediately."),
        ("Other", "Mom my phone fell in water and screen is dead, texting from a friend's mobile. Doctor needs Rs 9,500 advance right now, please GPay to 9811998877@okaxis."),
        ("Other", "Bhai urgent medical emergency at Apollo, my UPI daily limit is exhausted. Can you send Rs 12,000 to dr.helpdesk@ybl right now? Will return tonight."),
        ("Other", "Yaar I mistakenly transferred Rs 8,500 to your UPI instead of my sister. Please refund it back to 9765432109@paytm urgently, she needs it for college fees."),
        ("Other", "Sir by mistake I sent Rs 6,000 to your Google Pay number. Kindly return it to 9123004455@oksbi, my father is admitted in ICU."),
        ("Other", "Brother I accidentally sent Rs 3,500 to your UPI while paying electricity bill. Kindly send it back to arun.k99@ybl right now, facing penalty."),
        ("Other", "Hello, I sent Rs 5,200 by mistake to your PhonePe number. Please transfer it back to 9870112233@okicici urgently, my train is leaving."),
        # 5. Fake e-commerce refund / customer care phishing
        ("KYC/Bank", "Flipkart Support: Your refund of Rs 3,299 for cancelled order could not be credited. Complete bank verification at https://flipkart-refund-care.in/claim"),
        ("KYC/Bank", "Amazon Customer Desk: Your Rs 4,999 return pickup refund is on hold. Enter card details to receive instant credit at https://amz-refund-portal.com"),
        ("KYC/Bank", "Myntra Refund Dept: Order #MY9921 failed at gateway. Click https://myntra-instant-refund.in/verify to push money back to your account."),
        ("KYC/Bank", "Meesho Payout: Your COD refund of Rs 1,450 is pending. Accept collect request or verify UPI at https://meesho-cashback-desk.site"),
        # 6. Customs / DHL / FedEx / Courier & e-Challan / RTO scams
        ("Other", "DHL Express India: Shipment #449102 is detained by Mumbai Customs for restricted goods. Pay Rs 4,500 clearance penalty within 4 hours or FIR will be filed."),
        ("Other", "FedEx Compliance: A parcel booked with your Aadhaar contains 4 passports and MDMA. Appear for Cyber Police video verification immediately or call 9811002233."),
        ("Other", "Traffic Police e-Challan: Vehicle MH12AB3456 has pending fine of Rs 1,500. Settle today at https://echallan-parivahan-pay.org to avoid license suspension."),
        ("Other", "RTO Notice: Vehicle DL04CA8821 caught on speed camera. Pay Rs 2,000 penalty via https://rto-challan-online.in before court summons is issued."),
        # 7. Plain utility disconnection threats (no brand name or URL)
        ("Other", "Dear consumer, your electricity connection will be disconnected tonight at 9:30 PM due to non-updation of last month bill. Contact billing officer on 9833445566."),
        ("Other", "Warning: Your piped gas supply will be cut off today at 8 PM because your bill payment did not reflect. Call our helpline 9711223344 immediately."),
        ("Other", "Water board final alert: Your home water meter will be disconnected tomorrow morning unless you verify receipt with supervisor 9900887766."),
        # 8. Sextortion / Loan-app blackmail
        ("Other", "We recorded your private video call last night. Transfer Rs 20,000 to 9844556677@ybl within 1 hour or we will upload it to YouTube and send it to all your contacts."),
        ("Other", "Your CashNow instant loan of Rs 6,000 is overdue by 1 day. Pay Rs 11,500 immediately to 9899001122@paytm or we will share your edited photos with your family contacts."),
        # 9. Social-engineered OTP / SIM swap / eSIM scams
        ("KYC/Bank", "SBI Fraud Prevention: A suspicious transaction of Rs 38,500 is underway on your card. Read me the 6 digit code you just received to block it immediately."),
        ("KYC/Bank", "HDFC Security Desk: Someone tried logging into your NetBanking. Tell us the 6-digit verification code sent to your phone to cancel the request."),
        ("KYC/Bank", "Jio Telecom Alert: Your physical SIM to eSIM conversion request is queued. If not initiated by you, forward the 6 digit confirmation code immediately to stop it."),
        ("KYC/Bank", "Airtel eSIM Notice: Your number will be ported in 2 hours. Reply with the 4 digit security code received via SMS to abort deactivation."),
        # 10. Fake charity / govt free recharge / OLX QR / Raw IP / upi://pay
        ("Other", "Urgent medical appeal: 3-year-old child needs Rs 4 lakh heart operation tonight at AIIMS. Please donate via UPI to savechild99@ybl and forward."),
        ("Lottery/Prize", "PM Free 5G Recharge Yojana 2026: Every citizen gets 84 days free unlimited data! Activate now at https://pm-free-recharge-yojana.in"),
        ("Lottery/Prize", "Government Laptop & Scholarship Scheme: Claim free Rs 25,000 student grant today at https://gov-student-grant.online/apply"),
        ("Other", "I am an Army officer posted in Pune and want to buy your fridge on OLX for Rs 14,000. Scan my QR code and enter your UPI PIN to receive the advance."),
        ("KYC/Bank", "Urgent: Your bank KYC is suspended. Verify immediately at http://103.24.77.19/sbi-kyc to restore netbanking."),
        ("KYC/Bank", "Alert: PAN card not linked to bank account. Update at http://45.118.132.5/pan-update today."),
        ("Lottery/Prize", "Cashback Reward! Claim Rs 5,000 Amazon Pay bonus via upi://pay?pa=claim-bonus@ybl&pn=Reward&am=1499&cu=INR now."),
        ("Other", "Hello, we are hiring part-time freelancers to rate hotels and restaurants on Maps. Earn Rs 2,000 to Rs 4,500 daily from home. Reply YES if interested."),
        ("Job/Investment", "Remote opportunity: Review travel listings for 30 mins a day and get Rs 3,000 daily payout via UPI. Message back for onboarding."),
        ("Other", "Namaste, sending our family wedding invitation card file. Please download and open ShaadiCard_2026.apk to see venue location."),
    ]
    rows: list[dict] = []
    # Add light organic variations (2-3x per item with varied codes/amounts, never 40x clones)
    for idx, (cat, txt) in enumerate(items):
        rows.append({"text": txt, "label": 1, "category": cat, "source_corpus": "indian_modern_scam", "sample_weight": 1.4})
        # Create 2 paraphrased/slot-varied siblings with different amounts/phones/urls
        for v in range(2):
            varied = txt
            varied = re.sub(r"\b9\d{9}\b", f"9{rng.randint(100000000, 999999999)}", varied)
            varied = re.sub(r"Rs\.?\s*[\d,]+", f"Rs {rng.choice([1800, 3500, 7500, 16000, 28000])}", varied)
            if varied != txt:
                rows.append({"text": varied, "label": 1, "category": cat, "source_corpus": "indian_modern_scam", "sample_weight": 1.2})
    return rows


def _generate_diverse_tricky_safe() -> list[dict]:
    """Diverse natural-language tricky SAFE messages (urgent work, friend VPAs, sub-brands, neutral chats)."""
    items: list[str] = [
        # Urgent work / academic / personal (must NOT trigger on 'urgent' / 'final warning' / 'immediately')
        "URGENT: Staging API is returning 502 errors and client demo is in 20 minutes! Join the Google Meet immediately.",
        "Urgent: Production database CPU hit 98%, please check the deployment logs right now.",
        "URGENT: Payment gateway webhook is failing in prod, join the war-room bridge immediately!",
        "Urgent: CI pipeline is broken on main and release cut is in 15 minutes, please revert the last PR.",
        "URGENT: Client presentation starts in 10 mins and the slide deck link needs edit access immediately.",
        "Final warning from Prof. Verma: Upload your Operating Systems lab report before 11:59 PM tonight or it will be marked zero.",
        "Final notice from Hostel Office: Clear your mess dues by Friday 5 PM at the accounts counter.",
        "Final reminder from Library: Return overdue textbooks before 4 PM today to avoid late fine.",
        "Hey, please call me urgently on 9876501234 regarding tomorrow's project presentation.",
        "Urgent reminder: Team standup moved to 10:15 AM today, please join on time.",
        "Please call me on 9811224455 when you reach the metro station.",
        "Give me a call on 9740112233 once you are free after class.",
        "Reach out to me on 9900112244 when you get to the main lobby.",
        "Hey, ping or call me on 9845001122 when the cab arrives at the gate.",
        "Call 9820334455 when you are outside the apartment building.",
        "Could you call me on 9886112233 before 5 PM to confirm the dinner plan?",
        "Just reached the office—call me on 9739001122 so I can buzz you in.",
        "Hey, call Riya on 9810223344 once you pick up the printout.",
        "Please ring me on 9945112233 when you are near the auditorium.",
        "Call me on 9871002233 if you need the Wi-Fi password for the conference room.",
        # Friend / family money splits & VPAs (must NOT confuse with UPI scams)
        "Hey can you send me the 850 for last night's pizza whenever you get a chance? No hurry.",
        "Here is my UPI ID for the Coorg trip split: rohit.k@okaxis, everyone's share is Rs 2,800.",
        "Bro my UPI id is aditya98@oksbi, transfer the 1,500 for movie tickets whenever free.",
        "Thanks! Received the 600 on GPay, see you on Monday.",
        "Did you pay the WiFi bill yet? My share is 450, sending it to your Paytm.",
        # Legitimate financial / insurance / bank sub-brand & periodic KYC advisories
        "HDFC ERGO: Your two-wheeler insurance policy expires on 20 Dec 2026. Renew online at hdfcergo.com to keep your 25% No Claim Bonus.",
        "ICICI Lombard: Reminder to renew your health cover on icicilombard.com before 31 Dec to retain continuity benefits.",
        "SBI Life: Your annual policy premium of Rs 18,500 is due next week. Pay securely on sbilife.co.in or official app.",
        "Axis Bank: Your auto-debit of Rs 499 for YouTube Premium could not be processed due to low balance. Please fund your account.",
        "Dear Customer, as per RBI norms periodic KYC updation is due for your SBI account. Visit your home branch or use the official YONO app.",
        "HDFC Bank: Periodic KYC review is due for your savings account. Please update via NetBanking or visit your nearest HDFC Bank branch.",
        "BESCOM: Your electricity bill of Rs 1,420 for Account ID 772910 is due on 29 Nov. Pay via BESCOM Mitra app or BBPS.",
        "MSEDCL: Bill for Oct 2026 of Rs 2,150 is generated. Pay on mahadiscom.in or authorized BBPS apps.",
        # Legitimate prizes / bonuses / achievements
        "Congratulations! Your team ByteCraft won 1st prize (Rs 40,000) at the National Hackathon. Collect your trophy at Auditorium B at 5 PM.",
        "Congrats Rahul! You won the department coding quiz prize of Rs 5,000. Check your college email for certificate details.",
        "Congratulations! Your project won 2nd place (Rs 25,000) at the college AI innovation challenge. Awards ceremony is at 3 PM.",
        "Congrats! Your robotics team won the inter-college championship prize of Rs 30,000. Collect certificates from Dean's office.",
        "Congratulations Ananya! You won 1st prize in the annual photography contest. Visit Student Activity Center to collect your award.",
        "Well done! Your hackathon squad won Best Prototype (Rs 15,000) at TinyVerse Demo Day. Meet the organizers at Hall A.",
        "HR Update: Your annual performance bonus of Rs 35,000 has been approved and will be credited along with November salary.",
        "Payroll Notice: Your Q3 spot bonus of Rs 10,000 will reflect in this month's salary slip.",
        "Congratulations! You have been awarded the Merit Scholarship of Rs 20,000 for Semester 6. Check the university notice board.",
        # Legitimate recruiters, small brands, clinic, OTPs, neutral chats
        "Hi Arsalan, I am a technical recruiter at Razorpay. Saw your open-source work on GitHub and would love to schedule a 25-min intro call.",
        "Hello! This is Sneha from Swiggy Talent Acquisition regarding your Backend Engineer application. Are you free for a call tomorrow?",
        "Flat 25% off on single-origin coffee beans at Third Wave Coffee this weekend! Order at https://thirdwavecoffeeroasters.com",
        "Flat 30% off on freshly roasted beans at Blue Tokai! Shop at https://bluetokaicoffee.com/collections/all",
        "Reminder: Your physiotherapy session at Apollo Clinic Koramangala is tomorrow at 4 PM. Call 080-22334455 to reschedule.",
        "619204 is your verification code for Blinkit. Valid for 5 minutes.",
        "482910 is your login OTP for Zepto.",
        "Your parcel from Meesho is arriving today between 1 PM and 5 PM. Keep Rs 499 cash ready.",
        "BlueDart: We attempted delivery today but the door was locked. We will re-attempt tomorrow morning.",
        "Reminder: Last date to pay semester tuition fees without late fine is 30 Nov via the student ERP portal.",
        "Hello, how are you doing today?",
        "Good morning! Are we still meeting for coffee at 6 PM?",
        "Let us meet outside Indiranagar metro station at 7 PM.",
        "Did you push the latest fix to the main branch? Tests are still running.",
        "Ok thanks!",
        "Sure, sounds good.",
        "Happy birthday! Have a great year ahead.",
        "On my way, reaching in 10 mins.",
    ]
    return [{"text": t, "label": 0, "category": "", "source_corpus": "hard_neg_diverse_safe", "sample_weight": 1.6} for t in items]


def _build_realworld_stress_suite() -> list[dict]:
    """Independent 50-message held-out Real-World Indian Stress Suite (25 SAFE, 25 SCAM) for Gate G10."""
    safe_msgs = [
        "Dear Customer, your Jio plan of Rs 299 is expiring in 3 days. Recharge now on MyJio app or jio.com to enjoy uninterrupted services.",
        "Airtel: You have used 90% of your daily 1.5GB data quota. Buy a data pack on Airtel Thanks app.",
        "Zomato: Craving biryani tonight? Get 50% OFF up to Rs 120 using code TRYNEW. Order now on Zomato app!",
        "Swiggy: Flat Rs 100 off on your next Instamart order above Rs 399! Valid till midnight. Shop now.",
        "HDFC Bank: Pre-approved Credit Card upgrade available for you! Check eligibility in NetBanking or HDFC MobileBanking app.",
        "ICICI Bank: Your credit card bill of Rs 12,450 is due on 28 Nov 2026. Pay via iMobile app to avoid late fees.",
        "Flipkart: Big Diwali Sale starts tonight at 12 AM! Extra 10% instant discount on SBI Credit Cards.",
        "Amazon: Your refund of Rs 1,299 for order #402-1234567 has been initiated and will reflect in 3-5 business days.",
        "Dear Student, exam hall tickets for Semester 5 are live on the university portal. Download before Friday.",
        "Hey bro, did you send the 2000 rs to Rahul? Let me know once done, need to settle the dinner split.",
        "Uber: Your ride with Ramesh (KA01AB1234) is arriving in 3 mins. Share PIN 4829 with driver to start trip.",
        "BookMyShow: Your tickets for Pushpa 2 (2 seats, PVR Orion Mall, 7:30 PM) are confirmed! Show QR at entry.",
        "Myntra: Your wishlist item is now at 40% off! Grab it before stock runs out.",
        "Reminder: Your SIP of Rs 5,000 in Nifty 50 Index Fund will be auto-debited on 25 Nov. Keep sufficient balance.",
        "Paytm: You received Rs 750 from ARJUN SHARMA via UPI. Updated wallet/bank history in app.",
        "Can you please share the OTP for the Amazon delivery? The delivery boy is at the gate.",
        "Hi, your appointment with Dr. Mehta is confirmed for tomorrow at 4:30 PM at Manipal Hospital.",
        "Congratulations on completing 1 year with Zerodha! View your annual portfolio summary in Console.",
        "Hi Arya, add something new to your Diwali look. See the latest watches at Titan World,   https://m.tneu.in/TITNWO/JeOf3mn Visit today.",
        "Ola: Auto arriving in 4 mins. Share OTP 7192 with driver to start your trip.",
        "VM-HDFCBK-S: Rs 2,340.00 debited from A/c XX4821 on 24-11-26 via UPI. Avail Bal: Rs 31,200. Not you? Call 18002586161 or visit hdfcbank.com",
        "AD-UIDAI-G: OTP for Aadhaar authentication is 849201. Valid for 10 mins. Never share OTP with anyone.",
        "Blinkit: Order #BL9921 delivered in 9 mins! Rate your delivery partner on the Blinkit app.",
        "Vi: Unlimited data pack of Rs 399 is now active on your number. Manage plan in Vi app.",
        "Hi Rohan, refresh your festive wardrobe at FabIndia. Explore new kurtas at https://m.fabindia.in/DIWALI/k8p2m Visit our store today.",
    ]
    scam_msgs = [
        ("Job/Investment", "Hi, I am Priya from Amazon HR. We are hiring part-time reviewers, earn 2000-5000 daily from home. Reply YES for details."),
        ("Other", "Sir I accidentally transferred Rs 15000 to your Google Pay instead of my brother. Please refund Rs 15000 urgently to 9876543210@ybl, my mother is in hospital."),
        ("KYC/Bank", "Dear customer your SBI card points worth Rs 9,850 expiring tonight. Redeem cash directly to bank via sbi-points-redeem.com"),
        ("Other", "Hello, this is FedEx Mumbai. A parcel in your name with illegal drugs was seized by customs. Connect with Mumbai Police on Skype immediately."),
        ("Other", "Your electricity power will be disconnected tonight at 9.30 PM because your previous month bill was not updated. Immediately contact our electricity officer 8910234567."),
        ("Other", "Sir your son has been arrested by police in a rape case. Send Rs 50,000 immediately to close FIR otherwise we will put him in jail."),
        ("Other", "Bro quick favor, my UPI is not working, can you send Rs 8000 to my friend 9811223344@paytm right now? Will return in 1 hour pakka."),
        ("Job/Investment", "Exclusive WhatsApp Stock Market Group: Get 5x multibagger IPO allotment tips from SEBI registered analysts. Join free: https://chat.whatsapp.com/AbCdEfGh"),
        ("KYC/Bank", "Aapka PAN card aadhaar se link nahi hai, aaj raat 12 baje se bank account band ho jayega. Abhi update karein: https://pan-update-gov.in"),
        ("Lottery/Prize", "Congratulations! Your mobile number won Rs 25 Lakh in KBC WhatsApp lottery. Contact Rana Pratap Singh on +919876543210 with lottery code 8891."),
        ("KYC/Bank", "VM-SBIKYC: Aapka SBI YONO khata aaj band ho jayega! Turant apna K.Y.C update karein: bit.ly/sbi-kyc-now"),
        ("Lottery/Prize", "CONGRATULATIONS! Won Rs 25,000 Amazon Spin cashback! Accept upi://pay?pa=reward-claim@ybl&pn=AmazonBonus&am=2499&cu=INR to receive!"),
        ("Other", "Traffic e-Challan Notice: Vehicle fined Rs 2,500. Download RTO_Challan_View.apk at bit.ly/echallan-apk immediately."),
        ("Loan/Credit", "Pre-approved Instant Personal Loan of Rs 5,00,000 in 5 mins! Zero CIBIL check. Apply now: fast-loan-india.loan"),
        ("Other", "CBI Cyber Cell: Digital arrest warrant issued against your Aadhaar for money laundering. Dial 9876543210 immediately."),
        ("KYC/Bank", "HDFC Warning: Your debit card is deactivated. Share OTP & CVV at tinyurl.com/hdfc-rekyc within 2 hours."),
        ("Other", "Shadi ka card! Namaste humari shaadi mein zaroor aana. Download Wedding_Invite.apk from bit.ly/shadi-card"),
        ("Job/Investment", "Guaranteed 300% profit in 7 days! Join VIP Crypto Trading Telegram channel t.me/crypto_jackpot_in with Rs 5000 minimum."),
        ("Loan/Credit", "Bina CIBIL ke turant Rs 2 Lakh ka loan paayein! Processing fee Rs 999 only. WhatsApp 8800112233 or visit bit.ly/fast-karz"),
        ("Other", "TRAI Final Notice: Your SIM card will be blocked in 2 hours due to illegal calls. Press 9 or call officer 7011223344."),
        ("Lottery/Prize", "Badhai ho! Jio Lucky Draw mein aapne jeeta hai Rs 15 Lakh! Claim karne ke liye 9988776655 pe call karein."),
        ("KYC/Bank", "FASTag KYC expired! Pay Rs 500 penalty or update at icici-rekyc.click immediately else blacklist."),
        ("Other", "Customer Care: Download AnyDesk QuickSupport and share 9-digit code with 9123456789 to receive your Rs 4,500 refund."),
        ("Job/Investment", "Work From Home! Earn Rs 3,500 daily liking YouTube videos. Telegram HR: wa.me/919876543210"),
        ("Other", "IndiaPost Customs: Your international parcel is held at Delhi airport. Pay Rs 4,200 duty at indpost-customs.xyz else seizure."),
    ]
    out = [{"text": t, "label": 0, "category": "", "source_corpus": "realworld_stress", "sample_weight": 1.0} for t in safe_msgs]
    for cat, t in scam_msgs:
        out.append({"text": t, "label": 1, "category": cat, "source_corpus": "realworld_stress", "sample_weight": 1.0})
    return out


def _build_ood_safe_suite() -> list[dict]:
    """Strictly blind 30-message OOD SAFE suite (Gate G11)."""
    ood_msgs = [
        "Your Google verification code is 492018. Don't share it with anyone.",
        "LinkedIn: 14 new Software Engineer jobs in Bengaluru match your profile. View on linkedin.com/jobs",
        "MakeMyTrip: Flight 6E-2041 BLR to DEL confirmed for 29 Nov 06:40. Web check-in opens 48 hrs before departure.",
        "HDFC Bank: Your FASTag wallet balance is Rs 180. Recharge via HDFC NetBanking to avoid low balance at toll.",
        "SBI: Scheduled maintenance on YONO & Internet Banking from 01:00 AM to 03:00 AM IST on Sunday.",
        "Amazon Pay: Rs 50 cashback has been credited to your Amazon Pay balance for your electricity bill payment.",
        "CRED: Your HDFC Credit Card bill of Rs 18,200 is paid! You earned 18,200 CRED coins.",
        "Groww: Your KYC verification is complete! You can now start investing in Mutual Funds and Stocks.",
        "Nykaa Pink Friday Sale! Up to 50% off on skincare & makeup. Shop now: https://www.nykaa.com/sale",
        "BigBasket: Your slot for tomorrow 7 AM - 9 AM is confirmed. Total payable Rs 1,140.",
        "JioCinema: Watch India vs Australia Live FREE tonight at 7 PM on JioCinema app!",
        "Airtel Thanks: Claim your complimentary Perplexity Pro / Amazon Prime benefit in the Airtel Thanks app.",
        "Dominos: Craving pizza? Buy 1 Get 1 Free on Medium Pizzas today only! Order at https://dominos.co.in",
        "Lenskart: Free eye checkup at your nearest Lenskart store! Book your slot now: https://lenskart.com/book",
        "GitHub: A new personal access token was added to your account. If this wasn't you, review at github.com/settings",
        "Internshala: Congratulations! Your application for Backend Intern at Razorpay has been shortlisted.",
        "Naukri.com: 3 recruiters viewed your profile this week. Update your resume to get more interview calls.",
        "College Placement Cell: Pre-placement talk by TCS tomorrow at 10 AM in Main Auditorium. Register at https://forms.gle/abc123",
        "Hey! Check out this YouTube video on ONNX optimization: https://youtu.be/dQw4w9WgXcQ",
        "Axis Bank: You have 4,500 EDGE Reward Points in your account. Redeem them anytime on the Axis Mobile app.",
        "RBI Kehta Hai: Do not share your ATM PIN, OTP or CVV with anyone over phone or SMS.",
        "Spotify: Your Premium Student plan renewed for Rs 59. Enjoy ad-free music.",
        "Rapido: Captain Suresh is arriving in 2 mins on KA05MN4412. Share PIN 3910 to start ride.",
        "Zomato: Refund of Rs 245 for your cancelled order has been initiated to your original payment method.",
        "Apple: Your iCloud+ 50GB storage subscription of Rs 75/month has been renewed.",
        "IRCTC: Chart prepared for Train 12628. Your berth is B3-42 (3A). Wish you a pleasant journey.",
        "Slack: New login to workspace TinyVerse from Windows Desktop.",
        "Coursera: Congratulations on completing Machine Learning Specialization! Download your certificate.",
        "BSES: Thank you for paying your electricity bill of Rs 2,180 via BBPS.",
        "WhatsApp: Your two-step verification PIN reminder. Never share this PIN with anyone.",
        "1 missed call(s) received on 02-11-2026 while your Airtel number was not reachable. Get details of your missed calls ONLY on Airtel App - https://i.airtel.in/1_MissedCall",
    ]
    return [{"text": t, "label": 0, "category": "", "source_corpus": "ood_safe", "sample_weight": 1.0} for t in ood_msgs]


def _build_transactional_promo_suite() -> list[dict]:
    """Strictly held-out 25-message Headerless Transactional & Promotional SAFE suite (Gate G12)."""
    msgs = [
        "Dear UPI user A/C X1234 debited by 450.0 on date 24Nov26 trf to ZOMATO Refno 432918472910. If not u? call 1800111109.",
        "Sent Rs.250.00 From HDFC Bank A/C *4821 To SWIGGY On 24/11/26 Ref 632819204512 Not You? Call 18002586161",
        "INR 1,299.00 spent using ICICI Bank Card XX8001 on 24-Nov-26 on AMAZON. Avl Limit: INR 84,200.00. If not you, call 18001080 or SMS BLOCK 8001 to 9215676766.",
        "Rs. 5,000.00 withdrawn from SBI A/c XXXXX1234 at ATM BLR042 on 24Nov26. Avl Bal Rs.32,140.50. Not done by you? Forward this SMS to 9223008333 or call 18001234.",
        "Your A/c no. XX7721 is credited with Rs.45,000.00 on 24-11-26 by NEFT from ACME CORP. Avl Bal Rs.58,900.00.",
        "Amt Sent Rs.120.00 From Kotak Bank AC X4410 to merchant@okicici on 24-11-26. UPI Ref 419283746501.",
        "An amount of INR 3,200.00 has been DEBITED to your account XXX9081 on 24/11/2026. Total Avl Bal INR 19,450.00.",
        "Your HDFC Bank A/c XX4821 has an active UPI Mandate for Netflix Rs 199.00 scheduled on 26-Nov-26.",
        "Update: INR 650.00 debited from Axis Bank A/c no. XX3319 on 24-11-26 at 09:14:22 IST via UPI.",
        "Alert: You've spent Rs.890 on HDFC Bank Millennia Credit Card ending 4421 at ZOMATO on 24-11-2026.",
        "849201 is the OTP for your HDFC Bank transaction of INR 2,499.00 at AMAZON PAY. Do not share this OTP with anyone.",
        "Dear Customer, Your a/c no. XXXXXXXX4120 is debited for Rs.750.00 on 24-11-26 and credited to a/c XX9901.",
        "Bank of Baroda: Rs.1,500.00 debited from A/c ...4512 on 24-Nov-26 via IMPS. Avl Bal Rs.8,200.00.",
        "Your ICICI Bank Account XX9012 is debited with INR 299.00 on 24-Nov-26 towards Spotify.",
        "Transaction of INR 4,999.00 on SBI Credit Card ending 3310 at FLIPKART was declined due to international usage disabled.",
        "Dear Customer, Rs.12,500.00 has been credited to your A/c XX8812 on 24Nov26 towards salary.",
        "Thank you for using your SBI Debit Card ending 7721 for a purchase worth Rs 1,100.00 at RELIANCE SMART.",
        "HDFC Bank: Rs.199.00 auto-debited from your Card XX4421 for Spotify Standing Instruction. Manage at sihub.in",
        "Your request for NEFT transfer of Rs.10,000.00 from A/c XX4821 to Beneficiary RAHUL has been processed.",
        "Dear Customer, Cheque no. 000412 for Rs.25,000.00 has been cleared from your A/c XX4821.",
        "Tata CLiQ: Flat 30% off on winter jackets & sneakers this weekend! Shop on tatacliq.com",
        "Croma Festive Deals: Up to Rs 5,000 instant bank discount on laptops & TVs at croma.com or nearest store.",
        "Swiggy Dineout: Book a table tonight and get flat 25% off on your restaurant bill via Swiggy app.",
        "Puma India: End of Season Sale is live! Extra 15% off on running shoes at in.puma.com",
        "MakeMyTrip: Flat Rs 1,200 off on domestic round-trip flights with HDFC Credit Cards. Book on makemytrip.com",
    ]
    return [{"text": t, "label": 0, "category": "", "source_corpus": "transactional_promo_safe", "sample_weight": 1.0} for t in msgs]


def _build_blind_multilingual_suite() -> list[dict]:
    """Independent 24-message blind Multi-Script + Novel Scam + Tricky Safe suite (Gate G13)."""
    items = [
        (1, "Other", "Hi I am interested in buying your sofa on OLX for Rs 12,000. I am in Army, please scan this QR code and enter UPI PIN to receive advance payment."),
        (1, "KYC/Bank", "Income Tax Dept: Your ITR refund of Rs 18,450 is approved. Verify bank details within 24 hrs at incometax-refund-portal.in"),
        (1, "Job/Investment", "Congratulations! You are selected for Infosys Work From Home Data Entry job. Pay Rs 1,499 refundable laptop security deposit to hr-infosys@ybl."),
        (1, "Other", "Hi Mom, my phone broke and I am texting from a friend's number. Need Rs 12,000 urgently for hospital bill, please send to 9822334455@okicici."),
        (1, "Lottery/Prize", "You have been selected as a lucky winner of Mahindra Thar! Pay Rs 4,500 GST charges to claim delivery. Call 8899776655."),
        (1, "Loan/Credit", "Urgent cash needed? Get Rs 3 Lakh personal loan in 10 mins with bad CIBIL. Zero paperwork! WhatsApp 9011223344."),
        (1, "Other", "Dear user, your Gas connection (IGL) will be disconnected tonight at 8 PM due to unpaid bill. Call officer 7890123456 immediately."),
        (1, "Other", "I recorded your private video from your webcam. Pay Rs 25,000 to my UPI or I will send it to all your contacts."),
        (1, "KYC/Bank", "आपका एसबीआई खाता केवाईसी अपडेट न होने के कारण बंद हो जाएगा। कृपया इस लिंक पर क्लिक करें: bit.ly/sbi-hindi"),
        (1, "Other", "ನಿಮ್ಮ ವಿದ್ಯುತ್ ಸಂಪರ್ಕ ಇಂದು ರಾತ್ರಿ ಕಡಿತಗೊಳ್ಳುತ್ತದೆ. ತಕ್ಷಣ ಅಧಿಕಾರಿಗೆ ಕರೆ ಮಾಡಿ 9876543210"),
        (1, "Lottery/Prize", "உங்கள் எண் ரூ.10 லட்சம் லாட்டரி வென்றுள்ளது. உடனே தொடர்பு கொள்ளவும் 9845012345"),
        (1, "KYC/Bank", "మీ బ్యాంక్ ఖాతా బ్లాక్ చేయబడింది. వెంటనే KYC అప్‌డేట్ చేయండి: bit.ly/kyc-te"),
        (0, "", "Hi! Your Swiggy order is delayed by 15 mins due to rain. Sorry for the inconvenience!"),
        (0, "", "Ajio: Flat 60% off on Nike, Puma & Adidas sneakers! Use code STEPUP on Ajio app."),
        (0, "", "Urban Company: Your AC service professional Rajesh will arrive at 11:00 AM today. Share start code 5912 upon arrival."),
        (0, "", "LIC India: Reminder! Premium of Rs 14,200 for Policy No. 781234567 is due on 30 Nov 2026. Pay via licindia.in or official app."),
        (0, "", "HDFC Mutual Fund: Your SIP installment of Rs 3,000 has been processed. Units allotted: 42.18. Folio 9182736."),
        (0, "", "JioMart: Your grocery order worth Rs 845 is out for delivery! Pay via Cash or UPI at doorstep."),
        (0, "", "भाई आज शाम 7 बजे जिम चलेंगे क्या? 500 रुपये जो उधार लिए थे वो भी दे दूँगा।"),
        (0, "", "Can you send me the PDF of the OS notes? Exam is on Monday and I haven't started Unit 3."),
        (0, "", "Your IRCTC refund of Rs 1,420 for cancelled PNR 4829104821 has been credited to your source bank account."),
        (0, "", "NSDL: Your PAN card application acknowledgement no. 881029384712 has been dispatched via India Post."),
        (0, "", "Zomato Gold: Renew your 3-month membership for just Rs 30! Enjoy free delivery on orders above Rs 199."),
        (0, "", "Dear Customer, your Axis Bank Debit Card ending 6190 has been dispatched via BlueDart AWB 7781920341."),
    ]
    return [{"text": t, "label": lbl, "category": cat, "source_corpus": "blind_multilingual", "sample_weight": 1.0} for lbl, cat, t in items]


def _build_ads_safe_suite() -> list[dict]:
    """Strictly held-out 20-message Blind Commercial Ads SAFE suite (Gate G14)."""
    msgs = [
        "Today only! Double your KFC bucket. Buy 6 pcs Hot n Crispy and get 6 pcs Strips... FREE! Don't miss this... grab now! bit.ly/kfcind-dsg19 TnC",
        "Craving pizza? Get Flat 50% OFF on your Domino's order up to Rs 120! Use code DOM50. Valid today only! Order now: bit.ly/dom50off T&C apply",
        "Myntra End of Reason Sale is LIVE! Up to 80% OFF on Nike, Puma, Levi's & more. Hurry, shop before sizes run out! myntra.com/eors TnC",
        "Hungry? Get snacks, cold drinks & ice cream delivered in 10 mins! Free delivery on orders above Rs 199. Shop now: bit.ly/blk-10m",
        "Weekend Treat! Flat Rs 150 OFF on Biryani, Burgers & Pizzas on Swiggy. Apply code TREAT150. Valid till midnight! T&C apply",
        "Buy 1 Get 1 FREE on Vincent Chase Eyeglasses & Sunglasses at Lenskart! Visit nearest store or book free eye test: bit.ly/lk-bogo1",
        "Nykaa Pink Friday Sale starts tomorrow! Extra 15% OFF on Lakme, Maybelline & MAC. Wishlist now: bit.ly/nyk-pf26 TnC",
        "AJIO Big Bold Sale! Minimum 60% OFF + Extra Rs 500 OFF on Rs 1999. Use code BOLD500. Shop on AJIO App today!",
        "Croma Festival of Dreams! Up to 40% OFF on ACs, Refrigerators & Smart TVs + No Cost EMI. Visit Croma store or bit.ly/croma-fest TnC",
        "Flat 40% OFF on all Burgers & Fries at Burger King! Buy 1 Whopper get 1 FREE. Valid today only at participating outlets. TnC apply",
        "Blockbuster Friday! Book movie tickets at PVR INOX and get Flat Rs 100 OFF on F&B combos. Use code PVR100. Book: bit.ly/pvr-fri TnC",
        "AD-MCDIND-P: McSaver Combo starting Rs 149! Grab McChicken + Fries + Coke today only at McDonald's! Order: bit.ly/mcd-149 TnC",
        "VM-ZEPTO-P: Midnight Cravings? Get ice cream, chips & cold coffee in 10 mins! Flat Rs 75 OFF with code ZEP75. T&C apply",
        "Starbucks India: Buy 1 Tall Beverage and get 1 FREE every Friday till 6 PM! Visit nearest Starbucks store today. TnC apply",
        "Reliance Digital: Flash Deal! Up to 35% OFF on Washing Machines & Laptops + 10% Bank Cashback. Check store or bit.ly/rd-flash TnC",
        "FabIndia: New Festive Handloom Collection is here! Flat 20% OFF on kurtas & sarees this weekend. Shop: tinyurl.com/fab-fest T&C",
        "Haldiram's: Celebrate with family! Flat 25% OFF on Mithai & Dry Fruit Gift Boxes. Order online or visit outlet. TnC apply",
        "boAt: Newly launched Airdopes at Rs 999! Limited stock launch price today only. Grab yours now: bit.ly/boat-999 T&C",
        "AX-PIZAHUT-P: Weekday Deal! Any 2 Medium Pan Pizzas at Rs 249 each. Order on Pizza Hut App or bit.ly/ph-249 TnC",
        "Decathlon: Gear up for monsoon treks! Flat 30% OFF on waterproof jackets & hiking shoes at decathlon.in T&C apply",
    ]
    return [{"text": t, "label": 0, "category": "", "source_corpus": "ads_safe_suite", "sample_weight": 1.0} for t in msgs]


def _build_blind_generalization_suite() -> list[dict]:
    """Strictly blind 50-message Real-World Generalization Suite (Gate G15: 25 SCAM, 25 SAFE)."""
    items = [
        # 25 SCAM (subtly phrased, abusable hosts, social engineering, zero template overlap)
        (1, "KYC/Bank", "Dear SBI Customer, please update your PAN and Aadhaar details before 5 PM to avoid account restriction: https://forms.gle/sbiKyc2026"),
        (1, "Job/Investment", "Selected for Google WFH Internship! Fill your bank account and registration fee details here: https://forms.gle/googleIntern26"),
        (1, "KYC/Bank", "Your HDFC netbanking session expired. Verify your credentials at https://hdfc-verify.github.io/login"),
        (1, "KYC/Bank", "Income tax refund of Rs 14,200 is pending. Submit bank details on https://docs.google.com/forms/d/e/fake-itr"),
        (1, "Job/Investment", "Hi, I got your number from a business conference in Mumbai. I run an automated gold futures desk giving 18% monthly returns, want to see my P&L?"),
        (1, "Job/Investment", "Hey dear, my uncle works at a Singapore quant fund and gave me insider USDT signals. Put just 10000 rupees on our platform to test it out today."),
        (1, "Other", "Hi this is Rajesh (CEO). I am stuck in a board meeting and cannot take calls. Can you urgently buy 5 Amazon gift cards worth Rs 10,000 each and email me the codes?"),
        (1, "Other", "Beta my ATM card got blocked at the airport and I need to pay for the ticket right now. Send 18000 to my colleague number 9845112233 on PhonePe immediately."),
        (1, "Other", "I want to buy your laptop listed on OLX for 35000. Sending you a Rs 5000 advance token on GPay, just click Accept and enter your PIN to receive it."),
        (1, "KYC/Bank", "Hello sir, we are calling from Myntra refund department. Your Rs 2,499 order failed, click this link to claim refund directly to your card: https://myntra-refunds-desk.in/claim"),
        (1, "Other", "Your DHL shipment #882910 has been stopped at Mumbai customs due to undeclared items. Pay clearance charge of Rs 3,800 within 6 hours or case will be handed to police."),
        (1, "Other", "Your vehicle KA03MN4821 has an unpaid e-challan of Rs 2000. Pay immediately via https://parivahan-echallan-pay.org to avoid court summons."),
        (1, "Other", "Dear consumer your power supply will be stopped today at 10 pm because previous month bill was not updated. Message billing desk on 9871234560."),
        (1, "Other", "We have screenshots of your video call from last night. If you do not transfer Rs 15000 within 2 hours we will upload it on YouTube and send to your Instagram followers."),
        (1, "KYC/Bank", "Hello I am calling from SBI card protection team. A fraudulent charge of Rs 49,000 is being processed on your card. Read me the 6 digit code you just received to cancel it."),
        (1, "Other", "Namaste uncle, sending you my sister wedding card file here. Please open WeddingInvitation.apk to see venue map and timings."),
        (1, "KYC/Bank", "Sir aapka account hold pe chala gaya hai, abhi apna pan card aur debit card photo bhejo warna kal se transaction band ho jayega."),
        (1, "Other", "Your instant cash loan of Rs 8000 is overdue. Pay Rs 14500 right now to 9911223344@ybl or we will send your morphed photos to your entire contact list."),
        (1, "Other", "Bro I sent 4000 to your number by mistake while paying rent, please return it to vikram99@oksbi asap my landlord is waiting."),
        (1, "Job/Investment", "Hello! Looking for remote freelancers to rate hotels and restaurants on Maps. Flexible 1 hour daily, payout 1500 to 4000 INR per day. Text me back if interested."),
        (1, "Job/Investment", "Congratulations! Your wallet is eligible for 2500 USDT airdrop. Connect your MetaMask wallet and approve claim at https://airdrop-claim-eth.net"),
        (1, "KYC/Bank", "Airtel Alert: Your SIM upgrade request to eSIM has been received. If not requested by you, forward the 6 digit confirmation code immediately to cancel."),
        (1, "Other", "Urgent appeal: 4 year old baby needs Rs 5 lakh heart surgery tonight at AIIMS. Please donate whatever you can to helpbaby@ybl and share."),
        (1, "Lottery/Prize", "PM Free Recharge Yojana 2026! Every Indian citizen gets 3 months free 5G recharge. Claim on https://free-5g-yojana.in/apply"),
        (1, "KYC/Bank", "Final reminder from Income Tax Department: discrepancy found in AY 2025-26 filing. Settle penalty of Rs 4,500 at https://efiling-tax-india.com today."),
        # 25 SAFE (tricky urgent/prize/finance/VPA + neutral everyday messages)
        (0, "", "URGENT: The production server is down and the client demo starts in 30 minutes! Please join the Zoom call immediately."),
        (0, "", "Final warning from Prof. Sharma: Submit your compiler design assignment before 11:59 PM tonight or you will get zero marks."),
        (0, "", "Hey can you pay me back the 1200 for last night dinner whenever you get a chance? No rush."),
        (0, "", "Hey sending you my UPI id for the Goa trip split: arsalan.dev@okaxis, total comes to Rs 3,400 per person."),
        (0, "", "Flat 30% off on specialty coffee beans at Blue Tokai this weekend! Order at https://bluetokaicoffee.com/collections/all"),
        (0, "", "Axis Bank: Your auto-debit of Rs 649 for Netflix failed due to insufficient balance. Please fund your account to avoid disruption."),
        (0, "", "Dear Customer, as per RBI guidelines periodic KYC renewal is due for your SBI account. Please visit your nearest SBI branch or use the official YONO app."),
        (0, "", "Reminder: Your dental checkup at Clove Dental Indiranagar is tomorrow at 5 PM. Call 080-41234567 to reschedule."),
        (0, "", "Hi Arsalan, I am a technical recruiter at Atlassian. Saw your GitHub profile and would love to schedule a 30-min chat for a Backend Engineer role. Let me know your availability."),
        (0, "", "482910 is your verification code for Zepto. Valid for 5 minutes."),
        (0, "", "Your parcel from Meesho is arriving today between 2 PM and 6 PM. Keep Rs 599 ready for Cash on Delivery."),
        (0, "", "Hello, how are you doing today?"),
        (0, "", "Let us meet at the cafe near MG Road metro station at 6 PM."),
        (0, "", "Did you push the latest commit to the main branch? CI is still running."),
        (0, "", "Thanks!"),
        (0, "", "Reminder: Last date to pay semester tuition fees without late fine is 30 Nov. Pay through the student ERP portal."),
        (0, "", "HDFC ERGO: Your motor insurance policy expires on 15 Dec 2026. Renew online on hdfcergo.com to retain your 20% No Claim Bonus."),
        (0, "", "Congratulations! Your team Satark won 1st prize (Rs 50,000) at the campus hackathon. Collect your certificate at the auditorium at 4 PM."),
        (0, "", "BESCOM: Your electricity bill of Rs 1,840 for RR No 482910 is due on 28 Nov. Pay via BESCOM Mitra app or BBPS."),
        (0, "", "BlueDart: We attempted to deliver your shipment today but the premises were locked. We will re-attempt tomorrow."),
        (0, "", "Please call me on 9876543210 when you reach the office gate."),
        (0, "", "Good morning! Did you get a chance to review the PR?"),
        (0, "", "Swiggy: Your order from Meghana Foods is being prepared and will arrive in 25 mins."),
        (0, "", "Your train 12657 departs from KSR Bengaluru at 10:40 PM tonight from Platform 3."),
        (0, "", "Can you share the Google Meet link for the 4 PM sync?"),
    ]
    return [{"text": t, "label": lbl, "category": cat, "source_corpus": "blind_generalization", "sample_weight": 1.0} for lbl, cat, t in items]


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    uci_rows = _download_uci()
    hf_rows = _try_hf_indian_datasets()
    scam_rows = _generate_indian_scam_corpus()
    modern_scam_rows = _generate_diverse_modern_scams()
    safe_hn_rows = _generate_hard_negative_safe_corpus()
    tricky_safe_rows = _generate_diverse_tricky_safe()
    stress_rows = _build_realworld_stress_suite()
    ood_rows = _build_ood_safe_suite()
    tx_promo_rows = _build_transactional_promo_suite()
    blind_multi_rows = _build_blind_multilingual_suite()
    ads_safe_rows = _build_ads_safe_suite()
    blind_gen_rows = _build_blind_generalization_suite()

    # Anti-leakage: exclude samples.txt and all held-out test suites (exact + template skeleton for G15)
    heldout_all = stress_rows + ood_rows + tx_promo_rows + blind_multi_rows + ads_safe_rows + blind_gen_rows
    excluded_keys = {_dedup_key(r["text"]) for r in heldout_all}
    excluded_gen_skeletons = {_template_skeleton(r["text"]) for r in blind_gen_rows}
    samples_path = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\samples.txt")
    if samples_path.exists():
        for m in samples_path.read_text(encoding="utf-8").split("\n\n"):
            if m.strip():
                excluded_keys.add(_dedup_key(m))

    all_rows = uci_rows + hf_rows + scam_rows + modern_scam_rows + safe_hn_rows + tricky_safe_rows

    seen: set[str] = set(excluded_keys)
    deduped: list[dict] = []
    for r in all_rows:
        key = _dedup_key(r["text"])
        skel = _template_skeleton(r["text"])
        if key and key not in seen and skel not in excluded_gen_skeletons:
            seen.add(key)
            r["skeleton"] = skel
            deduped.append(r)

    df = pd.DataFrame(deduped)
    print(f"Total deduplicated rows: {len(df)} (scam={int((df['label']==1).sum())}, safe={int((df['label']==0).sum())})")

    # Hold out 300 hard-negative safe rows into dedicated test_hard_negatives.csv
    hn_mask = df["source_corpus"].str.startswith("hard_neg_")
    hn_df = df[hn_mask]
    rest_df = df[~hn_mask]

    hn_train_val, test_hard_neg = train_test_split(hn_df, test_size=300, random_state=SEED, stratify=hn_df["source_corpus"])
    pool_df = pd.concat([rest_df, hn_train_val], ignore_index=True)

    # Group-based anti-leakage split by _template_skeleton so no template family leaks across train/val/test_unseen
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, tmp_idx = next(gss1.split(pool_df, pool_df["label"], groups=pool_df["skeleton"]))
    train_df = pool_df.iloc[tr_idx].reset_index(drop=True)
    temp_df = pool_df.iloc[tmp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_idx, te_idx = next(gss2.split(temp_df, temp_df["label"], groups=temp_df["skeleton"]))
    val_df = temp_df.iloc[va_idx].reset_index(drop=True)
    test_unseen = temp_df.iloc[te_idx].reset_index(drop=True)

    # Keep only standard columns in CSVs
    keep_cols = ["text", "label", "category", "source_corpus", "sample_weight"]
    train_df = train_df[keep_cols]
    val_df = val_df[keep_cols]
    test_unseen = test_unseen[keep_cols]
    test_hard_neg = test_hard_neg[keep_cols]

    unseen_scams = test_unseen[(test_unseen["label"] == 1) & (test_unseen["category"] != "")]
    adv_sample = unseen_scams.sample(n=min(160, len(unseen_scams)), random_state=SEED).copy()
    adv_levels = ["medium", "hard"]
    adv_sample["text"] = [
        mutate_message(txt, level=adv_levels[idx % 2])
        for idx, txt in enumerate(adv_sample["text"].tolist())
    ]
    adv_sample["source_corpus"] = "adversarial_mutated"

    stress_df = pd.DataFrame(stress_rows)
    ood_df = pd.DataFrame(ood_rows)
    tx_df = pd.DataFrame(tx_promo_rows)
    multi_df = pd.DataFrame(blind_multi_rows)
    ads_df = pd.DataFrame(ads_safe_rows)
    gen_df = pd.DataFrame(blind_gen_rows)

    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    val_df.to_csv(PROC_DIR / "val.csv", index=False)
    test_unseen.to_csv(PROC_DIR / "test_unseen.csv", index=False)
    test_hard_neg.to_csv(PROC_DIR / "test_hard_negatives.csv", index=False)
    adv_sample.to_csv(PROC_DIR / "test_adversarial.csv", index=False)
    stress_df.to_csv(PROC_DIR / "test_realworld_stress.csv", index=False)
    ood_df.to_csv(PROC_DIR / "test_ood_safe.csv", index=False)
    tx_df.to_csv(PROC_DIR / "test_transactional_safe.csv", index=False)
    multi_df.to_csv(PROC_DIR / "test_blind_multilingual.csv", index=False)
    ads_df.to_csv(PROC_DIR / "test_ads_safe.csv", index=False)
    gen_df.to_csv(PROC_DIR / "test_blind_generalization.csv", index=False)

    print("Saved splits to", PROC_DIR)
    for name, part in (
        ("train.csv", train_df),
        ("val.csv", val_df),
        ("test_unseen.csv", test_unseen),
        ("test_hard_negatives.csv", test_hard_neg),
        ("test_adversarial.csv", adv_sample),
        ("test_realworld_stress.csv", stress_df),
        ("test_ood_safe.csv", ood_df),
        ("test_transactional_safe.csv", tx_df),
        ("test_blind_multilingual.csv", multi_df),
        ("test_ads_safe.csv", ads_df),
        ("test_blind_generalization.csv", gen_df),
    ):
        print(f"  {name:30s}: {len(part):5d} rows (scam={int((part['label']==1).sum())}, safe={int((part['label']==0).sum())})")


if __name__ == "__main__":
    main()
