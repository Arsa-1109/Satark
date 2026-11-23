/**
 * URL defanging and risk-badge helpers so users never accidentally tap/copy live phishing links.
 * Strictly avoids literal external URL strings to satisfy Gate G9 packaging checks.
 */

const SHORTENER_HOSTS = new Set([
  "bit.ly",
  "tinyurl.com",
  "t.co",
  "goo.gl",
  "ow.ly",
  "is.gd",
  "cutt.ly",
  "rb.gy",
  "shorturl.at",
  "tiny.cc",
]);

const OFFICIAL_SUFFIXES = [".gov.in", ".nic.in", ".bank.in", ".rbi.org.in", ".npci.org.in"];

export function defangUrl(url) {
  if (!url) return "";
  let out = url.replace(/^https?:\/\//i, (m) =>
    m.toLowerCase().startsWith("https") ? "hxxps://" : "hxxp://"
  );
  const protoIdx = out.indexOf("://");
  const prefix = protoIdx !== -1 ? out.slice(0, protoIdx + 3) : "";
  const rest = protoIdx !== -1 ? out.slice(protoIdx + 3) : out;
  const slashIdx = rest.indexOf("/");
  const host = slashIdx !== -1 ? rest.slice(0, slashIdx) : rest;
  const path = slashIdx !== -1 ? rest.slice(slashIdx) : "";
  return `${prefix}${host.replace(/\./g, "[.]")}${path}`;
}

export function classifyUrlRisk(url, triggers = []) {
  const lower = (url || "").toLowerCase();
  const withoutProto = lower.replace(/^[a-z]+:\/\//, "");
  const host = withoutProto.split("/")[0].split(":")[0];

  if (/\.apk(?:\?|$)/i.test(lower)) {
    return { label: "APK Download", tone: "risk" };
  }
  if (/^\d{1,3}(?:\.\d{1,3}){3}$/.test(host)) {
    return { label: "Raw IP Link", tone: "risk" };
  }
  if (host.includes("xn--")) {
    return { label: "Punycode Spoof", tone: "risk" };
  }
  if (SHORTENER_HOSTS.has(host)) {
    return { label: "Shortened Link", tone: "warn" };
  }
  const hasBrandMismatch = triggers.some(
    (t) => t.polarity === "risk" && /mismatch|spoof|suspicious/i.test(t.reason || "")
  );
  if (hasBrandMismatch) {
    return { label: "Brand Mismatch", tone: "risk" };
  }
  if (OFFICIAL_SUFFIXES.some((sfx) => host.endsWith(sfx))) {
    return { label: "Official Domain", tone: "safe" };
  }
  return { label: "Extracted Link", tone: "neutral" };
}
