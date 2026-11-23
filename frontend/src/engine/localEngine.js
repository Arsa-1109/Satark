/**
 * Layer 1 + Layer 2 Local JS Engine (mirrors normalizer.py, forensics.py, engine.py).
 * Runs 100% offline in browser / Service Worker PWA using window.__SATARK_MODEL__.
 */

const MAX_RAW_CHARS = 10000;
const TRUNCATE_CHARS = 5000;

const ZERO_WIDTH_RE = /[\u200b-\u200f\u202a-\u202e\u2060\ufeff]/g;

const CONFUSABLES = {
  "\u0430": "a", "\u0410": "A",
  "\u0435": "e", "\u0415": "E",
  "\u043e": "o", "\u041e": "O",
  "\u0440": "p", "\u0420": "P",
  "\u0441": "c", "\u0421": "C",
  "\u0445": "x", "\u0425": "X",
  "\u0443": "y", "\u0423": "Y",
  "\u0456": "i", "\u0406": "I",
  "\u043a": "k", "\u041a": "K",
  "\u043c": "m", "\u041c": "M",
  "\u0442": "t", "\u0422": "T",
  "\u043d": "h", "\u041d": "H",
  "\u0432": "b", "\u0412": "B",
  "\u03b1": "a", "\u0391": "A",
  "\u03b5": "e", "\u0395": "E",
  "\u03bf": "o", "\u039f": "O",
  "\u03c1": "p", "\u03a1": "P",
  "\u03ba": "k", "\u039a": "K",
  "\u03c4": "t", "\u03a4": "T",
  "\u03b9": "i", "\u0399": "I",
};

const SPACED_PART_RE = /\b([a-zA-Z](?:[.\-_\s]{1,2}[a-zA-Z]){2,7})\b/g;
const LEET_MAP = { "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s" };
const PURE_CODE_RE = /^(?:xx+\d+|\*+\d+|rs\.?\d+|\d+[a-z]{0,2}|[a-z]{1,3}\d{3,}|v\d+(?:\.\d+)*)$/i;
const TOKEN_RE = /[A-Za-z0-9@$]+/g;
const URL_SPAN_RE = /(?:https?:\/\/|upi:\/\/|www\.)\S+/gi;

const HINGLISH_MAP = [
  [/\b(?:paise|pese|paisaa)\b/gi, "paisa"],
  [/\b(?:khaata|khta)\b/gi, "khata"],
  [/\b(?:turnt|turanth)\b/gi, "turant"],
  [/\b(?:jita|jeete|jeeto)\b/gi, "jeeta"],
  [/\b(?:bhejein|bhejdo|bhejde)\b/gi, "bhejo"],
  [/\b(?:bnd|bandh)\b/gi, "band"],
  [/\b(?:apka|aapke|apke)\b/gi, "aapka"],
  [/\b(?:khatam|khtm)\b/gi, "khatam"],
];

function decodeLeetToken(token) {
  if (PURE_CODE_RE.test(token)) return [token, false];
  const hasAlpha = /[a-zA-Z]/.test(token);
  const hasLeet = /[013457@$]/.test(token);
  if (!(hasAlpha && hasLeet)) return [token, false];
  let out = "";
  for (const c of token) {
    out += LEET_MAP[c] ?? c;
  }
  return [out, out !== token];
}

export function normalizeText(text, validate = true) {
  if (typeof text !== "string") throw new Error("Input text must be a string.");
  if (validate) {
    if (!text.trim()) throw new Error("Input text cannot be empty or whitespace-only.");
    if (text.length > MAX_RAW_CHARS) throw new Error(`Input text exceeds ${MAX_RAW_CHARS} characters.`);
  }

  let truncated = false;
  let working = text;
  if (working.length > TRUNCATE_CHARS) {
    working = working.slice(0, TRUNCATE_CHARS);
    truncated = true;
  }

  const obfuscations = [];
  const stripped = working.replace(ZERO_WIDTH_RE, "");
  if (stripped !== working) obfuscations.push("zero_width");
  working = stripped.normalize("NFKC");

  let hadHomoglyph = false;
  let folded = "";
  for (const ch of working) {
    const m = CONFUSABLES[ch];
    if (m !== undefined) {
      folded += m;
      hadHomoglyph = true;
    } else {
      folded += ch;
    }
  }
  if (hadHomoglyph) obfuscations.push("cyrillic_homoglyph");
  working = folded;

  const collapsed = working.replace(SPACED_PART_RE, (m) => m.replace(/[.\-_\s]+/g, ""));
  if (collapsed !== working) obfuscations.push("spaced_token");
  working = collapsed;

  const urlSpans = [];
  for (const m of working.matchAll(URL_SPAN_RE)) {
    urlSpans.push([m.index, m.index + m[0].length]);
  }
  const inUrl = (idx) => urlSpans.some(([s, e]) => idx >= s && idx < e);

  let hadLeet = false;
  working = working.replace(TOKEN_RE, (tok, offset) => {
    if (inUrl(offset)) return tok;
    const [dec, changed] = decodeLeetToken(tok);
    if (changed) hadLeet = true;
    return dec;
  });
  if (hadLeet) obfuscations.push("leetspeak");

  let alphaCount = 0;
  let upperCount = 0;
  for (const c of working) {
    if ((c >= "a" && c <= "z") || (c >= "A" && c <= "Z")) {
      alphaCount++;
      if (c >= "A" && c <= "Z") upperCount++;
    }
  }
  if (alphaCount >= 15 && upperCount / alphaCount > 0.65) {
    obfuscations.push("excessive_caps");
  }

  const cleanText = working.replace(/[ \t]+/g, " ").trim();
  let canonical = cleanText.toLowerCase();
  for (const [pat, repl] of HINGLISH_MAP) {
    canonical = canonical.replace(pat, repl);
  }

  return { rawText: text, cleanText, canonicalText: canonical, obfuscations, truncated };
}

const SHORTENERS = new Set([
  "bit.ly", "tinyurl.com", "t.ly", "cutt.ly", "rb.gy",
  "is.gd", "ow.ly", "shorturl.at", "goo.gl", "rebrand.ly", "s.id",
]);

const SUSPICIOUS_TLDS = [
  ".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf", ".gq",
  ".buzz", ".vip", ".club", ".shop", ".online", ".site", ".live",
  ".icu", ".loan", ".work",
];

const OFFICIAL_DOMAINS = [
  "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com", "kotak.com",
  "amazon.in", "flipkart.com", "irctc.co.in", "swiggy.com", "zomato.com",
  "bluedart.com", "delhivery.com", "npci.org.in", "rbi.org.in",
  "onlinesbi.sbi", "pnbindia.in", "bankofbaroda.in", "uidai.gov.in",
  "epfindia.gov.in", "indiapost.gov.in", "cybercrime.gov.in",
  "tneu.in", "titan.co.in", "tanishq.co.in", "croma.com", "tatacliq.com",
  "myntra.com", "nykaa.com", "lenskart.com", "bookmyshow.com", "jiomart.com",
  "bigbasket.com", "blinkit.com", "zepto.in", "airtel.in", "jio.com", "vi.in",
  "makemytrip.com", "goibibo.com", "uber.com", "olacabs.com", "urbancompany.com",
  "dominos.co.in", "decathlon.in", "bsesdelhi.com", "bescom.co.in", "tatapower.com",
  "bseindia.com", "nseindia.com", "sebi.gov.in", "trai.gov.in", "dot.gov.in",
  "github.com", "linkedin.com", "naukri.com", "internshala.com", "zerodha.com",
  "groww.in", "cred.club", "jiocinema.com", "hotstar.com", "youtu.be",
  "youtube.com", "forms.gle", "google.com",
];

const BRAND_KEYWORDS = [
  "sbi", "hdfc", "icici", "axis", "pnb", "bob", "kotak",
  "paytm", "phonepe", "gpay", "rbi", "npci", "epfo", "uidai",
  "aadhaar", "indiapost", "bescom", "msedcl", "trai", "cbi",
];

const URGENCY_LEX = [
  "urgent", "immediately", "24 hours", "24 hrs", "blocked today", "expires today",
  "ends today", "deactivated today", "cut tonight", "expire",
  "suspended", "blocked", "deactivated", "last warning", "final notice",
  "turant", "abhi", "band", "jaldi", "aaj", "warn",
];
const THREAT_AUTH_LEX = [
  "cbi", "police", "arrest", "warrant", "court", "customs", "fedex",
  "narcotics", "money laundering", "trai", "disconnection", "legal action",
  "fir", "freeze", "seizure",
];
const REWARD_LEX = [
  "congratulations", "winner", "lottery", "kbc", "lucky draw", "jackpot",
  "prize", "won", "cashback", "free", "bonus", "spin", "jeeta", "inam", "gift",
];
const KYC_BANK_LEX = [
  "kyc", "pan", "aadhaar", "yono", "netbanking", "fastag", "cvv", "atm",
  "debit card", "credit card", "account block", "update kyc", "khata",
];
const LOAN_CREDIT_LEX = [
  "loan", "pre-approved", "cibil", "emi", "credit limit", "instant approval",
  "low interest", "processing fee", "karz", "sanction",
];
const JOB_INVEST_LEX = [
  "part time", "work from home", "daily income", "earn per day",
  "youtube like", "subscribe", "telegram task", "hr", "trading", "crypto",
  "stock tips", "guaranteed return", "investment", "salary",
];

const URL_EXTRACT_RE = /(?:https?:\/\/[^\s<>"']+|www\.[^\s<>"']+|\b[a-z0-9][a-z0-9-]{1,62}\.(?:com|in|co\.in|org\.in|gov\.in|nic\.in|bank\.in|fin\.in|net|org|ly|xyz|top|click|tk|ml|ga|cf|gq|buzz|vip|club|shop|online|site|live|icu|loan|work|at|gd|id|gy)(?:\/[^\s<>"']*)?)/gi;
const IPV4_RE = /^\d{1,3}(?:\.\d{1,3}){3}$/;
const APK_RE = /(?:\.apk\b|(?:download|install).{0,25}(?:app|apk|file))/i;
const REMOTE_APP_RE = /\b(?:anydesk|teamviewer|quicksupport|rustdesk|ammyy|screen\s*share)\b/i;
const UPI_URI_RE = /upi:\/\/pay\?[^\s<>"']+/gi;
const WA_TG_RE = /(?:wa\.me\/|chat\.whatsapp\.com\/|t\.me\/|\b(?:join|contact|message|dm)\b.{0,20}\b(?:whatsapp|telegram)\b|\btelegram\s+(?:group|channel|task|hr|bot|link|id)\b)/i;
const PHONE_CALLBACK_RE = /(?:\b(?:call|contact|whatsapp|helpline|dial)\b.{0,35}(?:\+91[-\s]?)?[6-9]\d{9}|(?:\+91[-\s]?)?[6-9]\d{9}.{0,35}\b(?:call|contact|whatsapp|helpline|dial)\b)/i;
const TOLLFREE_RE = /(?:1800[-\s]?\d{3}[-\s]?\d{4}|\b1930\b)/;
const CURRENCY_RE = /(?:(?:rs\.?|inr|₹)\s*(\d[\d,]*(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\s*(lakh|crore|k)\b)/gi;
const ASKS_OTP_RE = /\b(?:share|send|enter|tell|provide|verify|batao|bhejo)\b.{0,20}\b(?:otp|pin|cvv|mpin|password)\b/i;
const OTP_DO_NOT_SHARE_RE = /\b(?:do\s*not|don'?t|never|mat|pls\s*do\s*not)\b.{0,20}\b(?:share|disclose|tell|batana)\b(?:\s.{0,20}\b(?:otp|pin|cvv|code|password)\b)?/i;
const AWARENESS_PSA_RE = /\b(?:beware\s+(?:of|dealing)|unsolicited\s+tips|take\s+informed\s+decision|use\s+telecom\s+services\s+responsibly|official\s+and\s+legitimate|rbi\s+kehta\s+hai|report\s+cyber\s*fraud|cyber\s*dost|stay\s+safe\s+from\s+fraud)\b/i;
const PHYSICAL_HANDOVER_RE = /\b(?:driver|trip|cab|ride|uber|ola|rapido|delivery\s*(?:boy|partner|agent|executive)|doorstep|at\s+the\s+gate|amazon\s+delivery|flipkart\s+delivery)\b/i;
const UPI_VPA_RE = /\b[a-z0-9.\-_]{3,}@(?:ybl|okicici|okaxis|oksbi|okhdfcbank|paytm|ibl|axl|upi|apl)\b/i;
const DLT_HEADER_RE = /\b[A-Z]{2}-[A-Z0-9]{5,6}(?:-[TSGP])?\b/;
const DLT_TX_SUFFIX_RE = /\b[A-Z]{2}-[A-Z0-9]{5,6}-[TSG]\b/;
const BANK_TX_RE = /\b(?:debited|credited|spent|withdrawn)\b.{0,40}\b(?:a\/c|acct|account|card)\b.{0,20}(?:x{1,4}\d{3,4}|\*{2,4}\d{3,4})/i;
const BANK_TX_META_RE = /\b(?:avail|bal|balance|upi|imps|neft|ref)\b/i;
const DELIVERY_ACTION_RE = /\b(?:out for delivery|dispatched|delivered|awb|tracking|order\s*#)\b/i;
const DELIVERY_BRAND_RE = /\b(?:amazon|flipkart|swiggy|zomato|bluedart|delhivery|dtdc|ekart|myntra)\b/i;
const CASUAL_HINGLISH_RE = /\b(?:bhai|yaar|bro|kal|mummy|papa|ghar|milte|dedunga|deta hu|bhej de|ok|haha)\b/i;

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function lexDensity(textLower, lexicon) {
  let hits = 0;
  for (const term of lexicon) {
    if (new RegExp(`\\b${escapeRegExp(term)}\\b`).test(textLower)) hits++;
  }
  return Math.min(hits / 4.0, 1.0);
}

function extractHost(rawUrl) {
  try {
    const u = rawUrl.includes("://") ? rawUrl : `http://${rawUrl}`;
    return new URL(u).hostname.toLowerCase().replace(/\.$/, "");
  } catch {
    return "";
  }
}

function isOfficialHost(host) {
  if (!host) return false;
  if ([".bank.in", ".fin.in", ".gov.in", ".nic.in"].some((s) => host.endsWith(s))) return true;
  return OFFICIAL_DOMAINS.some((d) => host === d || host.endsWith("." + d));
}

export function parseUpiUris(text) {
  const results = [];
  const low = text.toLowerCase();
  const hasLure = ["receive", "cashback", "refund", "credit", "prize", "won", "claim"].some((w) => low.includes(w));
  for (const m of text.matchAll(UPI_URI_RE)) {
    const raw = m[0];
    try {
      const u = new URL(raw);
      const pa = u.searchParams.get("pa") || "";
      const pn = u.searchParams.get("pn") || "";
      const am = u.searchParams.get("am") || "";
      const cu = u.searchParams.get("cu") || "INR";
      const tn = u.searchParams.get("tn") || "";
      const riskNote = hasLure
        ? `Collect/Payment trap: paying ₹${am || "?"} to ${pa || "unknown VPA"} while claiming cashback/prize/refund (UPI PIN is never needed to receive money).`
        : `UPI payment request to ${pa || "unknown VPA"} for ${cu} ${am || "unspecified"}.`;
      results.push({ raw, pa, pn, am, cu, tn, risk_note: riskNote });
    } catch {
      // ignore malformed
    }
  }
  return results;
}

export function extractForensics(norm) {
  const clean = norm.cleanText;
  const canon = norm.canonicalText;
  const cleanLower = clean.toLowerCase();

  const triggers = [];
  const seen = new Set();
  const addTrigger = (span, reason, polarity) => {
    if (!span) return;
    const key = `${span.toLowerCase()}|${polarity}`;
    if (!seen.has(key)) {
      seen.add(key);
      triggers.push({ span, reason, polarity });
    }
  };

  const rawUrls = Array.from(clean.matchAll(URL_EXTRACT_RE), (m) => m[0].replace(/[.,);!]+$/, ""));
  const urls = Array.from(new Set(rawUrls));
  const hosts = urls.map(extractHost);

  const has_url = urls.length > 0 ? 1.0 : 0.0;
  const url_count = Math.min(urls.length, 5) / 5.0;
  const has_shortener = hosts.some((h) => SHORTENERS.has(h)) ? 1.0 : 0.0;
  const has_suspicious_tld = hosts.some((h) => SUSPICIOUS_TLDS.some((t) => h.endsWith(t))) ? 1.0 : 0.0;
  const has_raw_ip_url = hosts.some((h) => IPV4_RE.test(h)) ? 1.0 : 0.0;
  const has_punycode_url = hosts.some((h) => h.includes("xn--") || /[^\x00-\x7f]/.test(h)) ? 1.0 : 0.0;
  const has_official_domain = hosts.some(isOfficialHost) ? 1.0 : 0.0;

  urls.forEach((u, idx) => {
    const h = hosts[idx];
    if (SHORTENERS.has(h)) addTrigger(u, "URL shortener obscures destination", "risk");
    else if (IPV4_RE.test(h)) addTrigger(u, "Raw IPv4 URL instead of official domain", "risk");
    else if (h.includes("xn--") || /[^\x00-\x7f]/.test(h)) addTrigger(u, "Punycode/IDN lookalike domain", "risk");
    else if (SUSPICIOUS_TLDS.some((t) => h.endsWith(t))) addTrigger(u, "High-risk suspicious TLD", "risk");
    else if (isOfficialHost(h)) addTrigger(u, "Verified official Indian domain", "safe");
  });

  const waM = clean.match(WA_TG_RE);
  const has_wa_tg = waM ? 1.0 : 0.0;
  if (waM) addTrigger(waM[0], "Off-platform WhatsApp/Telegram lure", "risk");

  const mentionsBrand = BRAND_KEYWORDS.some((b) => new RegExp(`\\b${b}\\b`).test(canon));
  const brand_domain_mismatch = mentionsBrand && (has_url > 0 || has_wa_tg > 0) && has_official_domain === 0.0 ? 1.0 : 0.0;
  if (brand_domain_mismatch && urls.length > 0) addTrigger(urls[0], "Brand-domain mismatch (non-official link)", "risk");

  const apkM = clean.match(APK_RE);
  const negatedApk = /\b(?:avoid|never|do\s*not|don'?t)\b.{0,20}\b(?:download|install)/i.test(canon);
  const has_apk_lure = apkM && !negatedApk ? 1.0 : 0.0;
  if (has_apk_lure && apkM) addTrigger(apkM[0], "Malicious APK / sideloading lure", "risk");

  const remoteM = clean.match(REMOTE_APP_RE);
  const has_remote_access_app = remoteM ? 1.0 : 0.0;
  if (remoteM) addTrigger(remoteM[0], "Remote screen-share app request", "risk");

  const upi_payloads = parseUpiUris(clean);
  const has_upi_uri = upi_payloads.length > 0 ? 1.0 : 0.0;
  if (upi_payloads.length > 0) addTrigger(upi_payloads[0].raw.slice(0, 48), "Direct upi://pay payment request URI", "risk");

  const upiVpaM = clean.match(UPI_VPA_RE);
  const has_upi_vpa = upiVpaM ? 1.0 : 0.0;

  const hasUpiMention = has_upi_uri > 0 || /\b(?:upi\s*id|upi\s*pin)\b/.test(canon);
  const hasReceiveLure = ["receive", "cashback", "refund", "credit", "prize", "won", "claim"].some((w) =>
    new RegExp(`\\b${w}\\b`).test(canon)
  );
  const urgentTransferCues = [
    "accidentally", "mistakenly", "refund", "urgently", "urgent", "hospital",
    "emergency", "upi is not working", "send rs", "send ₹", "transfer rs",
    "right now", "return in", "cashback", "prize", "won", "claim",
  ];
  const hasUrgentTransferCue = urgentTransferCues.some((cue) => canon.includes(cue));
  const bankTxPre = clean.match(BANK_TX_RE);
  const isGenuineReceipt = Boolean(bankTxPre && BANK_TX_META_RE.test(clean)) || canon.includes("you received");
  const isUpiSafetyPsa = /\b(?:never\s+need\s+to\s+enter|never\s+ask|not\s+required\s+to\s+receive|safety\s+tip|npci)\b/i.test(canon);
  const urgentTransferTrap = has_upi_vpa > 0 && hasUrgentTransferCue && !isGenuineReceipt;

  const upi_collect_cashback_trap = ((hasUpiMention && hasReceiveLure) || urgentTransferTrap) && !isUpiSafetyPsa ? 1.0 : 0.0;
  if (urgentTransferTrap && upiVpaM) {
    addTrigger(upiVpaM[0], "Urgent / unverified UPI VPA transfer request", "risk");
  } else if (upi_collect_cashback_trap && upi_payloads.length === 0) {
    addTrigger("UPI / cashback", "UPI payment/PIN paired with receive/cashback claim", "risk");
  }

  const phoneM = clean.match(PHONE_CALLBACK_RE);
  const has_indian_phone_callback = phoneM ? 1.0 : 0.0;
  if (phoneM) addTrigger(phoneM[0].slice(0, 36), "Personal 10-digit mobile callback number", "risk");

  const tollM = clean.match(TOLLFREE_RE);
  const has_tollfree_official = tollM ? 1.0 : 0.0;
  if (tollM) addTrigger(tollM[0], "Official 1800/1930 helpline reference", "safe");

  let has_currency_amount = 0.0;
  let has_large_amount = 0.0;
  for (const cm of clean.matchAll(CURRENCY_RE)) {
    has_currency_amount = 1.0;
    if (cm[1]) {
      const cleaned = cm[1].replace(/,/g, "");
      if (cleaned && parseFloat(cleaned) >= 25000) has_large_amount = 1.0;
    } else if (cm[2] && cm[3]) {
      const unit = cm[3].toLowerCase();
      const num = parseFloat(cm[2]);
      if (unit === "lakh" || unit === "crore" || (unit === "k" && num >= 25)) has_large_amount = 1.0;
    }
  }

  const urgency_score = lexDensity(canon, URGENCY_LEX);
  const threat_authority_score = lexDensity(canon, THREAT_AUTH_LEX);
  const reward_lure_score = lexDensity(canon, REWARD_LEX);
  const kyc_bank_lex_score = lexDensity(canon, KYC_BANK_LEX);
  const loan_credit_lex_score = lexDensity(canon, LOAN_CREDIT_LEX);
  const job_invest_lex_score = lexDensity(canon, JOB_INVEST_LEX);

  for (const [lexList, reason] of [
    [URGENCY_LEX, "Artificial urgency / countdown pressure"],
    [THREAT_AUTH_LEX, "Authority / legal threat coercion"],
    [REWARD_LEX, "Reward / prize / cashback lure"],
  ]) {
    for (const term of lexList) {
      const m = cleanLower.match(new RegExp(`\\b${escapeRegExp(term)}\\b`));
      if (m) {
        addTrigger(m[0], reason, "risk");
        break;
      }
    }
  }

  const otpSafeM = canon.match(OTP_DO_NOT_SHARE_RE);
  const physicalM = canon.match(PHYSICAL_HANDOVER_RE);
  const asksOtpM = canon.match(ASKS_OTP_RE);
  const asks_otp_or_pin = asksOtpM && !otpSafeM && !physicalM ? 1.0 : 0.0;
  if (asks_otp_or_pin && asksOtpM) addTrigger(asksOtpM[0], "Solicits OTP / PIN / CVV", "risk");

  const awarenessM = canon.match(AWARENESS_PSA_RE);
  const noMaliciousTraps =
    has_shortener === 0 &&
    has_apk_lure === 0 &&
    has_suspicious_tld === 0 &&
    brand_domain_mismatch === 0 &&
    upi_collect_cashback_trap === 0 &&
    asks_otp_or_pin === 0;
  const has_otp_do_not_share = otpSafeM || (awarenessM && noMaliciousTraps) ? 1.0 : 0.0;
  if (otpSafeM) addTrigger(otpSafeM[0], "Standard anti-fraud OTP warning", "safe");
  else if (awarenessM && noMaliciousTraps) addTrigger(awarenessM[0], "Public cyber-safety / regulatory awareness advisory", "safe");

  const dltM = clean.match(DLT_HEADER_RE);
  const dlt_header = dltM ? dltM[0] : null;
  const has_dlt_header = dlt_header ? 1.0 : 0.0;
  const has_dlt_transactional_suffix = DLT_TX_SUFFIX_RE.test(clean) ? 1.0 : 0.0;
  if (dlt_header) addTrigger(dlt_header, "TRAI DLT registered sender header", "safe");

  const bankTxM = clean.match(BANK_TX_RE);
  const has_bank_tx_template = bankTxM && BANK_TX_META_RE.test(clean) ? 1.0 : 0.0;
  if (has_bank_tx_template && bankTxM) addTrigger(bankTxM[0].slice(0, 40), "Standard bank transaction alert template", "safe");

  const delivM = canon.match(DELIVERY_ACTION_RE);
  const has_delivery_template =
    delivM && DELIVERY_BRAND_RE.test(canon) && !["customs", "fee", "pay", "bit.ly", ".apk"].some((w) => canon.includes(w))
      ? 1.0
      : 0.0;
  if (has_delivery_template && delivM) addTrigger(delivM[0], "Genuine e-commerce/courier delivery pattern", "safe");

  const casualM = canon.match(CASUAL_HINGLISH_RE);
  const has_casual_hinglish_chat =
    casualM &&
    has_url === 0 &&
    has_apk_lure === 0 &&
    threat_authority_score === 0 &&
    has_upi_vpa === 0 &&
    upi_collect_cashback_trap === 0 &&
    asks_otp_or_pin === 0 &&
    has_indian_phone_callback === 0
      ? 1.0
      : 0.0;
  if (has_casual_hinglish_chat && casualM) addTrigger(casualM[0], "Informal Hinglish conversational marker", "safe");

  const obfuscation_count = Math.min(norm.obfuscations.length, 4) / 4.0;
  let alphaCnt = 0;
  let upperCnt = 0;
  let digitCnt = 0;
  let exclCnt = 0;
  for (const c of clean) {
    if ((c >= "a" && c <= "z") || (c >= "A" && c <= "Z")) {
      alphaCnt++;
      if (c >= "A" && c <= "Z") upperCnt++;
    } else if (c >= "0" && c <= "9") {
      digitCnt++;
    } else if (c === "!") {
      exclCnt++;
    }
  }
  const upper_ratio = alphaCnt >= 15 ? upperCnt / alphaCnt : 0.0;
  const digit_ratio = digitCnt / Math.max(clean.length, 1);
  const excl_ratio = Math.min(exclCnt, 5) / 5.0;
  const log_char_len = Math.min(Math.log1p(clean.length) / 8.5, 1.0);

  const features = [
    has_url, url_count, has_shortener, has_suspicious_tld, has_raw_ip_url,
    has_punycode_url, has_official_domain, brand_domain_mismatch, has_apk_lure, has_remote_access_app,
    has_upi_uri, upi_collect_cashback_trap, has_wa_tg, has_indian_phone_callback, has_tollfree_official,
    has_currency_amount, has_large_amount, urgency_score, threat_authority_score, reward_lure_score,
    kyc_bank_lex_score, loan_credit_lex_score, job_invest_lex_score, asks_otp_or_pin, has_otp_do_not_share,
    has_dlt_header, has_dlt_transactional_suffix, has_bank_tx_template, has_delivery_template, has_casual_hinglish_chat,
    obfuscation_count, upper_ratio, digit_ratio, excl_ratio, log_char_len,
  ];

  const round3 = (x) => Math.round(x * 1000) / 1000;
  const pillars = {
    urgency_fear: round3(Math.min(1.0, urgency_score * 0.65 + upper_ratio * 0.2 + excl_ratio * 0.15)),
    authority_spoof: round3(Math.min(1.0, threat_authority_score * 0.55 + brand_domain_mismatch * 0.45)),
    reward_lure: round3(Math.min(1.0, reward_lure_score * 0.65 + has_large_amount * 0.25 + upi_collect_cashback_trap * 0.2)),
    trap_vector: round3(
      Math.min(
        1.0,
        has_shortener * 0.3 + has_suspicious_tld * 0.25 + has_apk_lure * 0.45 + has_remote_access_app * 0.45 +
          has_upi_uri * 0.3 + asks_otp_or_pin * 0.4 + has_punycode_url * 0.35 + has_raw_ip_url * 0.35 +
          has_wa_tg * 0.2 + obfuscation_count * 0.25
      )
    ),
    safe_counter_signals: round3(
      Math.min(
        1.0,
        has_official_domain * 0.35 + has_dlt_transactional_suffix * 0.25 + has_otp_do_not_share * 0.3 +
          has_bank_tx_template * 0.3 + has_delivery_template * 0.25 + has_casual_hinglish_chat * 0.25 + has_tollfree_official * 0.15
      )
    ),
  };

  return { features, pillars, urls, upi_payloads, dlt_header, triggers: triggers.slice(0, 10) };
}

function buildVocabMap(vocabArr) {
  const m = new Map();
  for (let i = 0; i < vocabArr.length; i++) m.set(vocabArr[i], i);
  return m;
}

function vectorizeCharWb(canonical, vocabMap, idf) {
  const counts = new Map();
  const words = canonical.trim().split(/\s+/).filter(Boolean);
  for (const w of words) {
    const padded = ` ${w} `;
    for (let n = 3; n <= 5; n++) {
      for (let i = 0; i <= padded.length - n; i++) {
        const gram = padded.slice(i, i + n);
        const idx = vocabMap.get(gram);
        if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
      }
    }
  }
  const vec = new Map();
  let normSq = 0;
  for (const [idx, cnt] of counts.entries()) {
    const val = (1.0 + Math.log(cnt)) * idf[idx];
    vec.set(idx, val);
    normSq += val * val;
  }
  const norm = Math.sqrt(normSq);
  if (norm > 0) {
    for (const [idx, val] of vec.entries()) vec.set(idx, val / norm);
  }
  return vec;
}

function vectorizeWord(canonical, vocabMap, idf) {
  const tokens = canonical.match(/[\p{L}\p{N}_]+/gu) || [];
  const counts = new Map();
  for (let i = 0; i < tokens.length; i++) {
    const uIdx = vocabMap.get(tokens[i]);
    if (uIdx !== undefined) counts.set(uIdx, (counts.get(uIdx) || 0) + 1);
    if (i + 1 < tokens.length) {
      const bIdx = vocabMap.get(`${tokens[i]} ${tokens[i + 1]}`);
      if (bIdx !== undefined) counts.set(bIdx, (counts.get(bIdx) || 0) + 1);
    }
  }
  const vec = new Map();
  let normSq = 0;
  for (const [idx, cnt] of counts.entries()) {
    const val = (1.0 + Math.log(cnt)) * idf[idx];
    vec.set(idx, val);
    normSq += val * val;
  }
  const norm = Math.sqrt(normSq);
  if (norm > 0) {
    for (const [idx, val] of vec.entries()) vec.set(idx, val / norm);
  }
  return vec;
}

function plattProb(z, a, b) {
  const v = a * z + b;
  if (v >= 0) return 1.0 / (1.0 + Math.exp(-v));
  const ez = Math.exp(v);
  return ez / (1.0 + ez);
}

function forensicRiskScore(f) {
  const risk =
    f[2] * 0.25 + f[3] * 0.2 + f[4] * 0.3 + f[5] * 0.3 + f[7] * 0.35 +
    f[8] * 0.45 + f[9] * 0.4 + f[10] * 0.25 + f[11] * 0.35 + f[12] * 0.15 +
    f[17] * 0.25 + f[18] * 0.25 + f[19] * 0.25 + f[23] * 0.4 + f[30] * 0.2;
  const safe =
    f[6] * 0.3 + f[24] * 0.35 + f[26] * 0.15 + f[27] * 0.3 + f[28] * 0.25 + f[29] * 0.25;
  return Math.max(0.0, Math.min(1.0, 0.5 + (risk - safe)));
}

function inferSubType(cleanText, f, category) {
  const low = cleanText.toLowerCase();
  if (f[8] > 0 || low.includes(".apk")) return "Malicious APK";
  if (["digital arrest", "cbi", "narcotics", "money laundering", "warrant", "cyber cell"].some((w) => low.includes(w))) {
    return "Digital Arrest";
  }
  if (f[11] > 0 || f[10] > 0) return "UPI Collect Trap";
  if (["electricity", "bijli", "bescom", "msedcl", "disconnect", "trai", "customs", "parcel"].some((w) => low.includes(w))) {
    return "Utility Threat";
  }
  if (f[9] > 0 || f[7] > 0) return "Impersonation";
  if (category === "Other") return "Impersonation";
  return null;
}

function redactVictimPii(text) {
  return text
    .replace(/\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g, "[AADHAAR_REDACTED]")
    .replace(/\b[A-Z]{5}\d{4}[A-Z]\b/g, "[PAN_REDACTED]")
    .replace(/(\b(?:otp|pin|code|cvv)\b[\s:is-]*)(\d{4,6})\b/gi, "$1[OTP_REDACTED]")
    .replace(/(\b(?:a\/c|acct|account|card)\b[\s:#no.]*[x*]*?)(\d{4,16})\b/gi, "$1[ACCT_REDACTED]");
}

function build1930Draft(rawText, category, subType, urls, upiPayloads) {
  const ts = new Date().toLocaleString("en-IN");
  const redacted = redactVictimPii(rawText.trim());
  const catLabel = category || "Suspected Scam";
  const subLabel = subType ? ` (${subType})` : "";
  const urlLine = urls.length ? urls.join(", ") : "None detected";
  const vpas = upiPayloads.map((p) => p.pa).filter(Boolean);
  const upiLine = vpas.length ? vpas.join(", ") : "None detected";
  return (
    `Sanchar Saathi Chakshu / 1930 Cybercrime Report Draft\n` +
    `Timestamp: ${ts}\n` +
    `Detected Category: ${catLabel}${subLabel}\n` +
    `Suspect URLs: ${urlLine}\n` +
    `Suspect UPI VPAs: ${upiLine}\n` +
    `Redacted Message Excerpt:\n"""\n${redacted.slice(0, 600)}\n"""`
  );
}

let cachedMaps = null;

export function analyzeLocal(text, source = "text", modelBundle = null) {
  const bundle = modelBundle || (typeof window !== "undefined" ? window.__SATARK_MODEL__ : null);
  if (!bundle) throw new Error("Satark offline model bundle not loaded.");
  if (!cachedMaps || cachedMaps.bundle !== bundle) {
    cachedMaps = {
      bundle,
      charMap: buildVocabMap(bundle.char_vocab),
      wordMap: buildVocabMap(bundle.word_vocab),
    };
  }

  const t0 = performance.now();
  const norm = normalizeText(text, true);
  const fr = extractForensics(norm);
  const f = fr.features;

  const alnumMatches = norm.cleanText.match(/[A-Za-z0-9\u0900-\u0d7f]/g);
  if ((alnumMatches ? alnumMatches.length : 0) < 2 && f[0] === 0) {
    return {
      verdict: "SAFE",
      category: null,
      sub_type: null,
      confidence_pct: 55.0,
      scam_probability: 0.45,
      certainty_tier: "Borderline — Verify Manually",
      model_mode: "pwa_js",
      truncated: norm.truncated,
      latency_ms: Math.round((performance.now() - t0) * 100) / 100,
      normalised_text: norm.cleanText,
      obfuscations_detected: norm.obfuscations,
      pillars: fr.pillars,
      forensics: { urls: fr.urls, upi_payloads: fr.upi_payloads, dlt_header: fr.dlt_header, triggers: fr.triggers },
      report_draft_1930: null,
    };
  }

  const cVec = vectorizeCharWb(norm.canonicalText, cachedMaps.charMap, bundle.char_idf);
  const wVec = vectorizeWord(norm.canonicalText, cachedMaps.wordMap, bundle.word_idf);
  const cLen = bundle.char_vocab.length;
  const wOffset = cLen;
  const fOffset = cLen + bundle.word_vocab.length;
  const fScaled = f.map((val, i) => (val - bundle.scaler_mean[i]) / (bundle.scaler_scale[i] || 1.0));

  const dotSparse = (coef, intercept) => {
    let s = intercept;
    for (const [idx, val] of cVec.entries()) s += coef[idx] * val;
    for (const [idx, val] of wVec.entries()) s += coef[wOffset + idx] * val;
    for (let i = 0; i < fScaled.length; i++) s += coef[fOffset + i] * fScaled[i];
    return s;
  };

  const logitA = dotSparse(bundle.stage_a.coef, bundle.stage_a.intercept);
  let pScam = plattProb(logitA, bundle.stage_a.platt_a, bundle.stage_a.platt_b);

  if ((f[8] > 0 || f[7] > 0 || f[11] > 0 || f[23] > 0) && pScam < 0.65) {
    pScam = Math.max(pScam, 0.78);
  }

  const verdict = pScam >= bundle.stage_a.threshold ? "SCAM" : "SAFE";
  const confidencePct = Math.round(Math.max(pScam, 1.0 - pScam) * 1000) / 10;

  const pFor = forensicRiskScore(f);
  let certaintyTier = "High Certainty";
  if ((pScam >= 0.38 && pScam <= 0.68) || (Math.abs(pScam - pFor) >= 0.45 && pScam < 0.85)) {
    certaintyTier = "Borderline — Verify Manually";
  } else if (confidencePct < 82.0) {
    certaintyTier = "Moderate";
  }

  let category = null;
  let subType = null;
  let reportDraft = null;
  if (verdict === "SCAM") {
    const classes = bundle.stage_b.classes;
    const rawLogits = classes.map((_, ci) => dotSparse(bundle.stage_b.coef[ci], bundle.stage_b.intercept[ci]));
    const maxLogit = Math.max(...rawLogits);
    const exps = rawLogits.map((z) => Math.exp(z - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sumExp);

    let maxIdx = 0;
    let maxP = probs[0];
    let entropy = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > maxP) {
        maxP = probs[i];
        maxIdx = i;
      }
      const pClamped = Math.max(probs[i], 1e-12);
      entropy -= probs[i] * Math.log(pClamped);
    }
    const hNorm = entropy / Math.log(classes.length);
    category = maxP < 0.45 || hNorm > 0.82 ? "Other" : classes[maxIdx];
    subType = inferSubType(norm.cleanText, f, category);
    reportDraft = build1930Draft(norm.rawText, category, subType, fr.urls, fr.upi_payloads);
  }

  return {
    verdict,
    category,
    sub_type: subType,
    confidence_pct: confidencePct,
    scam_probability: Math.round(pScam * 10000) / 10000,
    certainty_tier: certaintyTier,
    model_mode: "pwa_js",
    truncated: norm.truncated,
    latency_ms: Math.round((performance.now() - t0) * 100) / 100,
    normalised_text: norm.cleanText,
    obfuscations_detected: norm.obfuscations,
    pillars: fr.pillars,
    forensics: { urls: fr.urls, upi_payloads: fr.upi_payloads, dlt_header: fr.dlt_header, triggers: fr.triggers },
    report_draft_1930: reportDraft,
  };
}
