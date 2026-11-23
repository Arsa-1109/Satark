/**
 * 100% offline bilingual (English / Hinglish) guidance dictionary for Satark verdicts.
 * Zero external URLs or network calls.
 */

export const UI_COPY = {
  en: {
    scamHeadline: "Likely Scam — Do Not Act on This Message",
    borderlineHeadline: "Suspicious / Unclear — Verify Before Acting",
    safeHeadline: "Looks Safe — No Scam Signals Found",
    whatToDoTitle: "What to do right now",
    reportBtn: "Report to 1930 / Chakshu",
    call1930Btn: "Call 1930",
    warnFamilyBtn: "Warn Family",
    copiedWarning: "Warning Copied",
    exampleHeading: "Try an example",
    emptyTitle: "Check any SMS, WhatsApp forward, or payment link",
    emptySubtitle: "Paste a message below — 100% private and analyzed offline on your device.",
  },
  hi: {
    scamHeadline: "Khatra: Yeh Scam Lag Raha Hai — Koi Action Na Lein",
    borderlineHeadline: "Shaq Hai — Pehle Official Source Se Verify Karein",
    safeHeadline: "Safe Lag Raha Hai — Koi Scam Signal Nahi Mila",
    whatToDoTitle: "Abhi kya karein",
    reportBtn: "1930 / Chakshu par Report Karein",
    call1930Btn: "1930 Call Karein",
    warnFamilyBtn: "Family ko Alert Karein",
    copiedWarning: "Alert Copy Ho Gaya",
    exampleHeading: "Example try karein",
    emptyTitle: "Koi bhi SMS, WhatsApp message ya payment link check karein",
    emptySubtitle: "Neeche message paste karein — 100% offline aur private, sirf aapke device par.",
  },
};

export function getVerdictHeadline(result, lang = "en") {
  const copy = UI_COPY[lang] || UI_COPY.en;
  if (!result) return "";
  if (result.certainty_tier?.startsWith("Borderline")) return copy.borderlineHeadline;
  if (result.verdict === "SCAM") return copy.scamHeadline;
  return copy.safeHeadline;
}

export function buildHumanSummary(result, lang = "en") {
  if (!result) return "";
  const isBorderline = result.certainty_tier?.startsWith("Borderline");
  const isScam = result.verdict === "SCAM";
  const triggers = result.forensics?.triggers || [];
  const topRisk = triggers.find((t) => t.polarity === "risk");
  const topSafe = triggers.find((t) => t.polarity === "safe");
  const cat = result.category || "";
  const sub = result.sub_type || "";

  if (!isScam && !isBorderline) {
    if (topSafe) {
      return lang === "hi"
        ? `Is message mein safe signals mile hain (${topSafe.reason}), aur koi fraud link ya pressure tactic nahi hai.`
        : `This message matches legitimate patterns (${topSafe.reason}) with no high-risk scam indicators.`;
    }
    return lang === "hi"
      ? "Is message mein koi suspicious link, fake urgency, ya UPI/OTP trap nahi mila."
      : "No suspicious links, fake urgency, or UPI/OTP traps were detected in this message.";
  }

  if (isBorderline) {
    return lang === "hi"
      ? "Signals mixed hain. Koi bhi link click karne ya paisa bhejne se pehle official app ya bank helpline se confirm karein."
      : "Signals are mixed. Confirm directly via your official bank app or helpline before clicking links or sending money.";
  }

  const reasonNote = topRisk ? ` (${topRisk.reason})` : "";
  if (sub.toLowerCase().includes("apk")) {
    return lang === "hi"
      ? `Yeh ek malicious Android app (.apk) install karwane ki koshish hai${reasonNote} — isse phone aur bank dono hack ho sakte hain.`
      : `Attempts to trick you into installing an unverified Android .apk app${reasonNote}, which can steal OTPs and banking access.`;
  }
  if (sub.toLowerCase().includes("digital arrest")) {
    return lang === "hi"
      ? `Fake police/CBI "Digital Arrest" dhamki hai${reasonNote}. Asli police video call par arrest ya paisa nahi maangti.`
      : `Impersonates law enforcement with a fake "Digital Arrest" threat${reasonNote}. Real agencies never arrest or demand money on video calls.`;
  }
  if (result.forensics?.upi_payloads?.length > 0) {
    return lang === "hi"
      ? `Isme UPI collect/cashback trap hai — accept karne par aapke account se paisa KATEGA, aayega nahi${reasonNote}.`
      : `Contains a deceptive UPI payment request — approving it will DEDUCT money from your account, not credit it${reasonNote}.`;
  }
  if (cat === "KYC/Bank") {
    return lang === "hi"
      ? `Account band hone ka darr dikhakar fake link ya details maangi ja rahi hain${reasonNote}.`
      : `Uses account-suspension fear to push an unverified KYC/banking action${reasonNote}.`;
  }
  if (cat === "Lottery/Prize") {
    return lang === "hi"
      ? `Fake inaam ya cashback ka lalach dekar paisa ya details nikalne ki koshish hai${reasonNote}.`
      : `Lures you with a fake prize or cashback offer to extract money or sensitive details${reasonNote}.`;
  }
  if (cat === "Job/Investment") {
    return lang === "hi"
      ? `Ghar baithe guaranteed kamai ya task scam ka lalach lag raha hai${reasonNote}.`
      : `Shows hallmarks of a fake part-time job or guaranteed-return investment trap${reasonNote}.`;
  }
  if (cat === "Loan/Credit") {
    return lang === "hi"
      ? `Instant loan ke naam par advance fee ya data theft ka khatra lag raha hai${reasonNote}.`
      : `Resembles a predatory or fake instant-loan lure targeting personal data or advance fees${reasonNote}.`;
  }
  return lang === "hi"
    ? `Is message mein high-risk fraud signals mile hain${reasonNote}.`
    : `Multiple high-risk social-engineering or phishing signals were detected${reasonNote}.`;
}

export function getSafetyChecklist(result, lang = "en") {
  if (!result) return [];
  const isBorderline = result.certainty_tier?.startsWith("Borderline");
  const isScam = result.verdict === "SCAM";
  const cat = result.category || "";
  const sub = (result.sub_type || "").toLowerCase();
  const hasUpi = (result.forensics?.upi_payloads?.length || 0) > 0;

  if (!isScam && !isBorderline) {
    return lang === "hi"
      ? [
          "Phir bhi kisi anjaan link par login karne se pehle URL dhyaan se check karein.",
          "OTP, PIN, ya CVV kabhi kisi ke saath share na karein — chahe woh bank se bole.",
          "Agar paise ka transaction ho toh apni official bank app mein balance verify karein.",
        ]
      : [
          "Even for safe messages, never share your OTP, UPI PIN, or card CVV with anyone.",
          "For bank alerts, verify balances directly inside your official bank app.",
          "Stay cautious if a follow-up call asks you to install any screen-sharing app.",
        ];
  }

  if (isBorderline) {
    return lang === "hi"
      ? [
          "Message mein diye gaye link ya phone number par abhi click/call na karein.",
          "Official website ya bank ke card ke peeche likhe helpline number se confirm karein.",
          "Kisi bhi unknown app ko screen-share ya SMS permission na dein.",
        ]
      : [
          "Do not tap links or call numbers directly from this message.",
          "Verify independently using the official app or the helpline on the back of your card.",
          "Do not share OTPs or approve any UPI collect request until confirmed.",
        ];
  }

  if (sub.includes("apk")) {
    return lang === "hi"
      ? [
          "Is .apk file ko bilkul download ya install na karein.",
          "Agar install ho gayi hai toh turant phone ka internet band karein aur app uninstall karein.",
          "1930 par call karein ya sancharsaathi.gov.in (Chakshu) par report karein.",
        ]
      : [
          "Do NOT download or install the .apk file — it can read your SMS and banking OTPs.",
          "If already installed, turn off mobile data/Wi-Fi immediately and uninstall it.",
          "Report the sender on 1930 or sancharsaathi.gov.in (Chakshu).",
        ];
  }

  if (sub.includes("digital arrest")) {
    return lang === "hi"
      ? [
          "Video call turant kaatein — CBI/Police kabhi video call par Digital Arrest nahi karti.",
          "Kisi bhi 'verification' ya 'bail' account mein paisa transfer na karein.",
          "1930 par call karein aur number ko block + report karein.",
        ]
      : [
          "Hang up immediately — Indian Police, CBI, and Customs never conduct 'Digital Arrests' on video calls.",
          "Never transfer money to any 'verification' or 'RBI escrow' account.",
          "Call 1930 immediately if you feel threatened or shared personal details.",
        ];
  }

  if (hasUpi) {
    return lang === "hi"
      ? [
          "UPI app mein 'Pay' ya PIN bilkul enter na karein — paisa lene ke liye PIN nahi lagta.",
          "Pending UPI collect request ko 'Decline' aur 'Report' karein.",
          "Agar paisa kat gaya hai toh turant 1930 par call karein.",
        ]
      : [
          "Do NOT enter your UPI PIN — you never need a PIN to receive money or cashback.",
          "Decline and block the sender inside your UPI app.",
          "If money was debited, call 1930 immediately (golden hour recovery).",
        ];
  }

  if (cat === "KYC/Bank") {
    return lang === "hi"
      ? [
          "Message ke link par click na karein — KYC sirf official bank app ya branch mein hoti hai.",
          "OTP, Aadhaar, PAN, ya ATM PIN kisi ko na batayein.",
          "Sender ko block karein aur 1930 / Chakshu par report karein.",
        ]
      : [
          "Do NOT open the link — banks never ask for KYC updates via shortened links.",
          "Check your account status only in your official banking app or branch.",
          "Never share OTP, PAN, or Aadhaar details; report the message on 1930 / Chakshu.",
        ];
  }

  return lang === "hi"
    ? [
        "Is message mein diye gaye kisi bhi link par click na karein aur reply na karein.",
        "Koi bhi paisa, processing fee, ya OTP bilkul share na karein.",
        "1930 par call karein ya sancharsaathi.gov.in (Chakshu) par report karein.",
      ]
    : [
        "Do NOT click any links, download files, or reply to this sender.",
        "Never pay an advance fee, 'security deposit', or share any OTP/UPI PIN.",
        "Report the sender via 1930 or sancharsaathi.gov.in (Chakshu).",
      ];
}

export function buildFamilyWarningText(result, lang = "en") {
  const cat = result?.category || "Scam";
  const sub = result?.sub_type ? ` (${result.sub_type})` : "";
  if (lang === "hi") {
    return `Satark Alert: Yeh ${cat}${sub} scam message lag raha hai. Is tarah ke link par click na karein, koi .apk install na karein, aur OTP/UPI PIN share na karein. Zaroorat padne par 1930 par call karein.`;
  }
  return `Satark Alert: Be careful — this looks like a ${cat}${sub} scam message. Do not click the link, install any file, or share OTP/UPI PIN. If someone is affected, call 1930 immediately.`;
}
