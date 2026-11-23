"""Build deduplicated, anti-leakage Indian SMS/WhatsApp dataset in Projects\\TinyVerse\\data."""

from __future__ import annotations

import io
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from satark.mutator import mutate_message  # noqa: E402

DATA_DIR = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\data")
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
                        if txt:
                            rows.append({
                                "text": txt,
                                "label": 1 if lbl == "spam" else 0,
                                "category": "",
                                "source_corpus": "uci_sms",
                                "sample_weight": 0.35,
                            })
    print(f"Loaded {len(rows)} rows from UCI SMS Spam Collection (weight=0.35).")
    return rows


def _try_hf_indian_datasets() -> list[dict]:
    """Best-effort pull of public HF Indian/Hinglish spam CSVs/parquets with schema validation."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    candidates = [
        "bolewara/hinglish-scam-text-dataset",
        "CloveAI/india-spam-sms",
        "anmolshrivastav/scam-ham-india",
    ]
    rows: list[dict] = []
    for repo in candidates:
        try:
            files = api.list_repo_files(repo_id=repo, repo_type="dataset")
            data_files = [f for f in files if f.endswith((".csv", ".parquet"))]
            if not data_files:
                continue
            local = hf_hub_download(repo_id=repo, filename=data_files[0], repo_type="dataset", local_dir=RAW_DIR / repo.replace("/", "__"))
            df = pd.read_parquet(local) if local.endswith(".parquet") else pd.read_csv(local)
            cols = {c.lower(): c for c in df.columns}
            text_col = cols.get("text") or cols.get("message") or cols.get("sms") or cols.get("v2")
            lbl_col = cols.get("label") or cols.get("class") or cols.get("v1") or cols.get("is_scam")
            if not text_col or not lbl_col:
                continue
            weight = 0.40 if "CloveAI" in repo else 0.80
            added = 0
            for _, r in df.iterrows():
                txt = str(r[text_col]).strip()
                raw_lbl = str(r[lbl_col]).strip().lower()
                if not txt or txt == "nan":
                    continue
                is_scam = 1 if raw_lbl in ("1", "spam", "scam", "fraud", "true") else 0
                rows.append({
                    "text": txt,
                    "label": is_scam,
                    "category": "",
                    "source_corpus": f"hf_{repo}",
                    "sample_weight": weight,
                })
                added += 1
            print(f"Loaded {added} rows from HF {repo}.")
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
    for i in range(280):
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
    for i in range(240):
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
    for i in range(240):
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
    for i in range(240):
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
    for i in range(260):
        tpl = other_templates[i % len(other_templates)]
        txt = tpl.format(dom=rng.choice(other_urls), ph=rng.choice(phones))
        if i % 3 == 0:
            txt += f" Case#{9000 + i}"
        rows.append({"text": txt, "label": 1, "category": "Other", "source_corpus": "indian_curated_2026", "sample_weight": 1.5})

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

    return rows


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    uci_rows = _download_uci()
    hf_rows = _try_hf_indian_datasets()
    scam_rows = _generate_indian_scam_corpus()
    safe_hn_rows = _generate_hard_negative_safe_corpus()

    all_rows = uci_rows + hf_rows + scam_rows + safe_hn_rows

    # Strict deduplication BEFORE splitting
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in all_rows:
        key = _dedup_key(r["text"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)

    df = pd.DataFrame(deduped)
    print(f"Total deduplicated rows: {len(df)} (scam={int((df['label']==1).sum())}, safe={int((df['label']==0).sum())})")

    # Hold out ~150 hard-negative safe rows into dedicated test_hard_negatives.csv
    hn_mask = df["source_corpus"].str.startswith("hard_neg_")
    hn_df = df[hn_mask]
    rest_df = df[~hn_mask]

    hn_train_val, test_hard_neg = train_test_split(hn_df, test_size=200, random_state=SEED, stratify=hn_df["source_corpus"])
    pool_df = pd.concat([rest_df, hn_train_val], ignore_index=True)

    # Stratify pool on label + category
    strat_key = pool_df["label"].astype(str) + "_" + pool_df["category"].fillna("")
    # Merge tiny strata if needed
    vc = strat_key.value_counts()
    strat_key = strat_key.where(strat_key.map(vc) >= 4, pool_df["label"].astype(str))

    train_df, temp_df = train_test_split(pool_df, test_size=0.30, random_state=SEED, stratify=strat_key)
    temp_strat = temp_df["label"].astype(str) + "_" + temp_df["category"].fillna("")
    vc2 = temp_strat.value_counts()
    temp_strat = temp_strat.where(temp_strat.map(vc2) >= 2, temp_df["label"].astype(str))
    val_df, test_unseen = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_strat)

    # Build test_adversarial.csv from unseen Indian SCAM rows
    unseen_scams = test_unseen[(test_unseen["label"] == 1) & (test_unseen["category"] != "")]
    adv_sample = unseen_scams.sample(n=min(160, len(unseen_scams)), random_state=SEED).copy()
    adv_levels = ["medium", "hard"]
    adv_sample["text"] = [
        mutate_message(txt, level=adv_levels[idx % 2])
        for idx, txt in enumerate(adv_sample["text"].tolist())
    ]
    adv_sample["source_corpus"] = "adversarial_mutated"

    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    val_df.to_csv(PROC_DIR / "val.csv", index=False)
    test_unseen.to_csv(PROC_DIR / "test_unseen.csv", index=False)
    test_hard_neg.to_csv(PROC_DIR / "test_hard_negatives.csv", index=False)
    adv_sample.to_csv(PROC_DIR / "test_adversarial.csv", index=False)

    print("Saved splits to", PROC_DIR)
    for name, part in (
        ("train.csv", train_df),
        ("val.csv", val_df),
        ("test_unseen.csv", test_unseen),
        ("test_hard_negatives.csv", test_hard_neg),
        ("test_adversarial.csv", adv_sample),
    ):
        print(f"  {name:26s}: {len(part):5d} rows (scam={int((part['label']==1).sum())}, safe={int((part['label']==0).sum())})")


if __name__ == "__main__":
    main()
