/**
 * Private on-device scan history stored in localStorage.
 * Redacts victim PII (Aadhaar, PAN, OTP, Account numbers) before persisting previews.
 */

const STORAGE_KEY = "satark_scan_history_v1";
const MAX_ITEMS = 20;

const AADHAAR_RE = /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g;
const PAN_RE = /\b[A-Z]{5}\d{4}[A-Z]\b/g;
const OTP_RE = /\b(otp|pin|code|cvv)\b([\s:is\-]*)(\d{4,6})\b/gi;
const ACCT_RE = /\b(a\/c|acct|account|card)\b([\s:#no.]*[x*]*)(\d{4,16})\b/gi;

export function redactPreviewPii(text) {
  if (!text) return "";
  return text
    .replace(AADHAAR_RE, "[AADHAAR]")
    .replace(PAN_RE, "[PAN]")
    .replace(OTP_RE, "$1$2[OTP]")
    .replace(ACCT_RE, "$1$2[ACCT]");
}

export function loadHistory() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveHistory(items) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)));
  } catch {
    // ignore quota or privacy-mode storage errors
  }
}

export function addHistoryEntry({ rawText, sender, result, source }) {
  if (!rawText || !rawText.trim() || !result) return loadHistory();
  const cleanPreview = redactPreviewPii(rawText.trim()).replace(/\s+/g, " ");
  const entry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    timestamp: new Date().toISOString(),
    preview: cleanPreview.slice(0, 140),
    rawText: rawText.trim(),
    sender: sender || "",
    source: source || "text",
    verdict: result.verdict,
    certainty_tier: result.certainty_tier || "",
    category: result.category || null,
    sub_type: result.sub_type || null,
    confidence_pct: result.confidence_pct,
  };

  const existing = loadHistory();
  // Deduplicate consecutive identical rawText+sender entries
  const filtered = existing.filter(
    (item) => !(item.rawText === entry.rawText && (item.sender || "") === entry.sender)
  );
  const next = [entry, ...filtered].slice(0, MAX_ITEMS);
  saveHistory(next);
  return next;
}

export function removeHistoryEntry(id) {
  const next = loadHistory().filter((item) => item.id !== id);
  saveHistory(next);
  return next;
}

export function clearHistory() {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
  return [];
}
