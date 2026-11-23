import { useEffect, useState } from "react";
import { Layers, Link2, Swords, FileWarning, Copy, Check, PhoneCall, Eye, EyeOff } from "lucide-react";
import { cn } from "../utils/cn";
import { defangUrl, classifyUrlRisk } from "../utils/urlDefang";

export function ForensicDrawers({ result, mutationData, requestedTab, requestedTabNonce, drawerRef }) {
  const [tab, setTab] = useState("xray");
  const [copied, setCopied] = useState(false);
  const [showRawUrls, setShowRawUrls] = useState(false);

  useEffect(() => {
    if (requestedTab) {
      setTab(requestedTab);
    }
  }, [requestedTab, requestedTabNonce]);

  if (!result) return null;

  const pillars = result.pillars || {};
  const urls = result.forensics?.urls || [];
  const upis = result.forensics?.upi_payloads || [];
  const triggers = result.forensics?.triggers || [];
  const draft = result.report_draft_1930;

  const tabs = [
    { id: "xray", label: "4-Pillar X-Ray", icon: Layers },
    { id: "links", label: `Links & UPI (${urls.length + upis.length})`, icon: Link2 },
    ...(mutationData ? [{ id: "mutate", label: "Red-Team Diff", icon: Swords }] : []),
    ...(draft ? [{ id: "report", label: "1930 / Chakshu Draft", icon: FileWarning }] : []),
  ];

  const activeTab = tabs.some((t) => t.id === tab) ? tab : "xray";

  const handleCopyDraft = async () => {
    if (!draft) return;
    await navigator.clipboard.writeText(draft);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <section ref={drawerRef} className="rounded-xl border border-[#E4E4E0] bg-white">
      <div
        role="tablist"
        aria-label="Forensic breakdown"
        className="flex flex-wrap border-b border-[#E4E4E0] px-2 pt-2"
      >
        {tabs.map((t) => {
          const Icon = t.icon;
          const active = activeTab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setTab(t.id)}
              className={cn(
                "inline-flex min-h-[44px] items-center gap-1.5 border-b-2 px-3 py-2 text-xs font-semibold transition-colors duration-150 sm:min-h-[38px]",
                active
                  ? "border-[#111111] text-[#111111]"
                  : "border-transparent text-[#52525B] hover:text-[#111111]"
              )}
            >
              <Icon className="size-3.5" />
              {t.label}
            </button>
          );
        })}
      </div>

      <div className="p-4 sm:p-5">
        {activeTab === "xray" && (
          <div className="grid gap-3 sm:grid-cols-2">
            {[
              { key: "urgency_fear", label: "Urgency & Fear", tone: "bg-[#C81E1E]" },
              { key: "authority_spoof", label: "Authority / Brand Spoof", tone: "bg-[#C81E1E]" },
              { key: "reward_lure", label: "Reward / Greed Lure", tone: "bg-[#B45309]" },
              { key: "trap_vector", label: "Trap Vector (Link / APK / UPI / OTP)", tone: "bg-[#C81E1E]" },
              { key: "safe_counter_signals", label: "Safe Counter-Signals (DLT / Official)", tone: "bg-[#047857]" },
            ].map((item) => {
              const val = Math.round((pillars[item.key] || 0) * 100);
              return (
                <div key={item.key} className="rounded border border-[#E4E4E0] bg-[#F7F7F5] p-3">
                  <div className="flex items-center justify-between text-xs font-semibold text-[#111111]">
                    <span>{item.label}</span>
                    <span className="font-mono-tabular">{val}%</span>
                  </div>
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[#E4E4E0]">
                    <div className={cn("h-full", item.tone)} style={{ width: `${val}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === "links" && (
          <div className="space-y-3">
            {upis.length === 0 && urls.length === 0 && (
              <p className="text-xs text-[#52525B]">No URLs or upi://pay payloads detected in this message.</p>
            )}
            {upis.map((u, i) => (
              <div key={i} className="rounded border border-[#C81E1E]/30 bg-[#C81E1E]/5 p-3">
                <div className="text-xs font-bold text-[#C81E1E]">Decoded upi://pay Payload</div>
                <div className="font-mono-tabular mt-1 grid gap-1 text-xs text-[#111111] sm:grid-cols-2">
                  <span>Payee VPA (pa): {u.pa || "—"}</span>
                  <span>Payee Name (pn): {u.pn || "—"}</span>
                  <span>Amount (am): {u.cu} {u.am || "—"}</span>
                  <span>Note (tn): {u.tn || "—"}</span>
                </div>
                <p className="mt-2 text-xs font-medium text-[#C81E1E]">{u.risk_note}</p>
              </div>
            ))}
            {urls.length > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-[#52525B]">
                  Links are defanged by default so they cannot be opened accidentally.
                </span>
                <button
                  type="button"
                  onClick={() => setShowRawUrls((v) => !v)}
                  className="inline-flex min-h-[32px] items-center gap-1 rounded border border-[#E4E4E0] bg-[#F7F7F5] px-2 py-1 text-[11px] font-semibold text-[#111111]"
                >
                  {showRawUrls ? <EyeOff className="size-3" /> : <Eye className="size-3" />}
                  {showRawUrls ? "Defang" : "Show raw"}
                </button>
              </div>
            )}
            {urls.map((url, i) => {
              const risk = classifyUrlRisk(url, triggers);
              const badgeTone =
                risk.tone === "risk"
                  ? "border-[#C81E1E]/30 bg-[#C81E1E]/10 text-[#C81E1E]"
                  : risk.tone === "warn"
                  ? "border-[#B45309]/30 bg-[#B45309]/10 text-[#B45309]"
                  : risk.tone === "safe"
                  ? "border-[#047857]/30 bg-[#047857]/10 text-[#047857]"
                  : "border-[#E4E4E0] bg-white text-[#52525B]";
              return (
                <div
                  key={i}
                  className="flex flex-wrap items-center justify-between gap-2 rounded border border-[#E4E4E0] bg-[#F7F7F5] px-3 py-2"
                >
                  <span className="font-mono-tabular break-all text-xs text-[#111111]">
                    {showRawUrls ? url : defangUrl(url)}
                  </span>
                  <span className={cn("shrink-0 rounded border px-2 py-0.5 text-[11px] font-semibold", badgeTone)}>
                    {risk.label}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === "mutate" && mutationData && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="font-semibold text-[#111111]">
                Adversarial Resilience Check (Leetspeak + Spaced Tokens + Cyrillic Homoglyphs)
              </span>
              <span className="font-mono-tabular rounded border border-[#E4E4E0] bg-[#F7F7F5] px-2 py-0.5 font-semibold">
                Delta: {mutationData.confidence_delta >= 0 ? `+${mutationData.confidence_delta}` : mutationData.confidence_delta}%
              </span>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded border border-[#E4E4E0] bg-[#F7F7F5] p-3">
                <div className="text-[11px] font-bold uppercase text-[#52525B]">Original</div>
                <div className="mt-1 text-xs font-semibold text-[#111111]">
                  {mutationData.original.verdict} ({mutationData.original.confidence_pct}%)
                </div>
              </div>
              <div className="rounded border border-[#E4E4E0] bg-[#F7F7F5] p-3">
                <div className="text-[11px] font-bold uppercase text-[#B45309]">Mutated Attack</div>
                <div className="mt-1 text-xs font-semibold text-[#111111]">
                  {mutationData.mutated.verdict} ({mutationData.mutated.confidence_pct}%)
                </div>
                <p className="font-mono-tabular mt-2 break-words text-[11px] text-[#52525B]">
                  {mutationData.mutated_text}
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === "report" && draft && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-semibold text-[#111111]">
                PII-Redacted Incident Draft (OTP / Aadhaar / PAN / Account Masked)
              </span>
              <div className="flex flex-wrap items-center gap-2">
                <a
                  href="tel:1930"
                  className="inline-flex min-h-[36px] items-center gap-1 rounded border border-[#C81E1E]/40 bg-[#C81E1E]/10 px-2.5 py-1 text-xs font-semibold text-[#C81E1E]"
                >
                  <PhoneCall className="size-3.5" />
                  Call 1930
                </a>
                <button
                  type="button"
                  onClick={handleCopyDraft}
                  className="inline-flex min-h-[36px] items-center gap-1 rounded border border-[#E4E4E0] bg-[#F7F7F5] px-2.5 py-1 text-xs font-semibold text-[#111111] hover:bg-[#EAEAE5]"
                >
                  {copied ? <Check className="size-3.5 text-[#047857]" /> : <Copy className="size-3.5" />}
                  {copied ? "Copied" : "Copy Draft"}
                </button>
              </div>
            </div>
            <pre className="font-mono-tabular overflow-x-auto rounded border border-[#E4E4E0] bg-[#F7F7F5] p-3 text-xs leading-relaxed text-[#111111]">
              {draft}
            </pre>
            <p className="text-[11px] text-[#52525B]">
              Submit this draft at <span className="font-semibold text-[#111111]">sancharsaathi.gov.in (Chakshu)</span> for suspected fraud SMS/calls, or <span className="font-semibold text-[#111111]">cybercrime.gov.in</span> / helpline <span className="font-semibold text-[#111111]">1930</span> if money was lost.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
