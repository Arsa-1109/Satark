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
    s = text.lower()
    s = re.sub(r"https?://\S+|www\.\S+", "<URL>", s)
    s = re.sub(r"\d+", "#", s)
    return re.sub(r"\s+", " ", s).strip()


def _try_hf_indian_datasets() -> list[dict]:
    """Best-effort pull of clean public HF SMS/spam datasets with schema validation (excludes noisy CloveAI)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    candidates = [
        ("Deysi/spam-detection-dataset", 0.20),
        ("anmolshrivastav/scam-ham-india", 0.80),
    ]
    rows: list[dict] = []
    for repo, weight in candidates:
        try:
            files = api.list_repo_files(repo_id=repo, repo_type="dataset")
            data_files = [f for f in files if f.endswith((".csv", ".parquet"))]
            if not data_files:
                continue
            added = 0
            for df_file in data_files[:2]:
                local = hf_hub_download(repo_id=repo, filename=df_file, repo_type="dataset", local_dir=RAW_DIR / repo.replace("/", "__"))
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
                    is_scam = 1 if raw_lbl in ("1", "spam", "scam", "fraud", "true") else 0
                    rows.append({
                        "text": txt,
                        "label": is_scam,
                        "category": "",
                        "source_corpus": f"hf_{repo}",
                        "sample_weight": weight,
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
    ]
    hr_names = ["Priya", "Anjali", "Ritu", "Neha", "Sneha", "Kavita"]
    hr_comps = ["Amazon", "Flipkart", "Google Maps", "Meesho", "Nykaa"]
    lookalikes = ["sbi-points-redeem.com", "hdfc-reward-cash.in", "icici-points-claim.com", "axis-kyc-portal.in"]
    cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad"]
    for i in range(180):
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

    # 7. Telecom & DTH usage / plan expiry alerts (directing to official apps/domains)
    telecom_templates = [
        "Dear Customer, your Jio plan of Rs {amt} is expiring in {days} days. Recharge now on MyJio app or jio.com to enjoy uninterrupted services.",
        "Airtel: You have used {pct}% of your daily {gb}GB data quota. Buy a data pack on Airtel Thanks app.",
        "Vi: Your unlimited 5G pack validity ends in {days} days. Renew easily via Vi app or vi.in.",
        "BSNL: Your FTTH broadband bill of Rs {amt} is ready. Pay securely on bsnl.co.in or MyBSNL app.",
        "Tata Play: Your DTH balance is low (Rs 45). Recharge via Tata Play app to continue watching your channels.",
    ]
    for i in range(90):
        tpl = telecom_templates[i % len(telecom_templates)]
        txt = tpl.format(amt=rng.choice([239, 299, 349, 666, 799]), days=rng.randint(1, 5), pct=rng.choice([50, 90, 100]), gb=rng.choice(["1.5", "2.0", "2.5"]))
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

    return rows


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
    ]
    return [{"text": t, "label": 0, "category": "", "source_corpus": "ood_safe", "sample_weight": 1.0} for t in ood_msgs]


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    uci_rows = _download_uci()
    hf_rows = _try_hf_indian_datasets()
    scam_rows = _generate_indian_scam_corpus()
    safe_hn_rows = _generate_hard_negative_safe_corpus()
    stress_rows = _build_realworld_stress_suite()
    ood_rows = _build_ood_safe_suite()

    # Anti-leakage: exclude samples.txt, test_realworld_stress.csv, and test_ood_safe.csv
    excluded_keys = {_dedup_key(r["text"]) for r in stress_rows + ood_rows}
    samples_path = Path(r"C:\Users\arsal\Desktop\Projects\TinyVerse\samples.txt")
    if samples_path.exists():
        for m in samples_path.read_text(encoding="utf-8").split("\n\n"):
            if m.strip():
                excluded_keys.add(_dedup_key(m))

    all_rows = uci_rows + hf_rows + scam_rows + safe_hn_rows

    seen: set[str] = set(excluded_keys)
    deduped: list[dict] = []
    for r in all_rows:
        key = _dedup_key(r["text"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)

    df = pd.DataFrame(deduped)
    print(f"Total deduplicated rows: {len(df)} (scam={int((df['label']==1).sum())}, safe={int((df['label']==0).sum())})")

    # Hold out 300 hard-negative safe rows into dedicated test_hard_negatives.csv
    hn_mask = df["source_corpus"].str.startswith("hard_neg_")
    hn_df = df[hn_mask]
    rest_df = df[~hn_mask]

    hn_train_val, test_hard_neg = train_test_split(hn_df, test_size=300, random_state=SEED, stratify=hn_df["source_corpus"])
    pool_df = pd.concat([rest_df, hn_train_val], ignore_index=True)

    strat_key = pool_df["label"].astype(str) + "_" + pool_df["category"].fillna("")
    vc = strat_key.value_counts()
    strat_key = strat_key.where(strat_key.map(vc) >= 4, pool_df["label"].astype(str))

    train_df, temp_df = train_test_split(pool_df, test_size=0.30, random_state=SEED, stratify=strat_key)
    temp_strat = temp_df["label"].astype(str) + "_" + temp_df["category"].fillna("")
    vc2 = temp_strat.value_counts()
    temp_strat = temp_strat.where(temp_strat.map(vc2) >= 2, temp_df["label"].astype(str))
    val_df, test_unseen = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_strat)

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

    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    val_df.to_csv(PROC_DIR / "val.csv", index=False)
    test_unseen.to_csv(PROC_DIR / "test_unseen.csv", index=False)
    test_hard_neg.to_csv(PROC_DIR / "test_hard_negatives.csv", index=False)
    adv_sample.to_csv(PROC_DIR / "test_adversarial.csv", index=False)
    stress_df.to_csv(PROC_DIR / "test_realworld_stress.csv", index=False)
    ood_df.to_csv(PROC_DIR / "test_ood_safe.csv", index=False)

    print("Saved splits to", PROC_DIR)
    for name, part in (
        ("train.csv", train_df),
        ("val.csv", val_df),
        ("test_unseen.csv", test_unseen),
        ("test_hard_negatives.csv", test_hard_neg),
        ("test_adversarial.csv", adv_sample),
        ("test_realworld_stress.csv", stress_df),
        ("test_ood_safe.csv", ood_df),
    ):
        print(f"  {name:26s}: {len(part):5d} rows (scam={int((part['label']==1).sum())}, safe={int((part['label']==0).sum())})")


if __name__ == "__main__":
    main()
