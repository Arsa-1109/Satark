import { useState } from "react";
import { ShieldAlert, ShieldCheck, AlertTriangle, Clock, PhoneCall, FileWarning, Share2, Check } from "lucide-react";
import { cn } from "../utils/cn";
import {
  UI_COPY,
  getVerdictHeadline,
  buildHumanSummary,
  getSafetyChecklist,
  buildFamilyWarningText,
} from "../data/guidance";

export function VerdictHero({ result, verdictRef, lang = "en", labMode = false, onOpenReport }) {
  const [warnCopied, setWarnCopied] = useState(false);

  if (!result) return null;

  const copy = UI_COPY[lang] || UI_COPY.en;
  const isScam = result.verdict === "SCAM";
  const isBorderline = result.certainty_tier?.startsWith("Borderline");
  const showActions = isScam || isBorderline;

  const bandClass = isBorderline
    ? "bg-[#B45309]/[0.08] border-b border-[#B45309]/20"
    : isScam
    ? "bg-[#C81E1E]/[0.07] border-b border-[#C81E1E]/20"
    : "bg-[#047857]/[0.07] border-b border-[#047857]/20";

  const badgeClass = isBorderline
    ? "bg-[#B45309] text-white"
    : isScam
    ? "bg-[#C81E1E] text-white"
    : "bg-[#047857] text-white";

  const barClass = isBorderline
    ? "bg-[#B45309]"
    : isScam
    ? "bg-[#C81E1E]"
    : "bg-[#047857]";

  const headline = getVerdictHeadline(result, lang);
  const summary = buildHumanSummary(result, lang);
  const checklist = getSafetyChecklist(result, lang);

  const displayPct = isBorderline
    ? Math.min(100, Math.max(0, result.scam_probability * 100))
    : Math.min(100, Math.max(0, result.confidence_pct));
  const metricLabel = isBorderline
    ? copy.borderlineRiskLabel
    : isScam
    ? copy.scamConfidenceLabel
    : copy.safeConfidenceLabel;

  const handleWarnFamily = async () => {
    const msg = buildFamilyWarningText(result, lang);
    if (navigator.share) {
      try {
        await navigator.share({ title: "Satark Scam Alert", text: msg });
        return;
      } catch {
        // Fallback to clipboard if user cancels or share fails
      }
    }
    try {
      await navigator.clipboard.writeText(msg);
      setWarnCopied(true);
      setTimeout(() => setWarnCopied(false), 2000);
    } catch {
      // ignore
    }
  };

  return (
    <section
      ref={verdictRef}
      role="status"
      aria-live="polite"
      className="overflow-hidden rounded-xl border border-[#E4E4E0] bg-white"
    >
      <div className={cn("p-4 sm:p-5", bandClass)}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className={cn("flex size-10 shrink-0 items-center justify-center rounded-md", badgeClass)}>
              {isBorderline ? (
                <AlertTriangle className="size-5" />
              ) : isScam ? (
                <ShieldAlert className="size-5" />
              ) : (
                <ShieldCheck className="size-5" />
              )}
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-lg font-extrabold tracking-tight text-[#111111] sm:text-xl">
                  {headline}
                </h2>
                {result.category && (
                  <span className="rounded border border-[#E4E4E0] bg-white px-2 py-0.5 text-xs font-semibold text-[#111111]">
                    {result.category}
                  </span>
                )}
                {result.sub_type && (
                  <span className="rounded border border-[#C81E1E]/30 bg-white px-2 py-0.5 text-xs font-semibold text-[#C81E1E]">
                    {result.sub_type}
                  </span>
                )}
                {result.truncated && (
                  <span className="rounded border border-[#B45309]/40 bg-white px-2 py-0.5 text-[11px] font-medium text-[#B45309]">
                    Truncated to 5,000 chars
                  </span>
                )}
              </div>
              <p className="mt-1.5 text-sm leading-snug text-[#111111]">
                {summary}
              </p>
              {labMode && (
                <p className="mt-1 text-xs font-medium text-[#52525B]">
                  Certainty: <span className="text-[#111111]">{result.certainty_tier}</span>
                  {" · "}
                  <span className="font-mono-tabular">p_scam = {result.scam_probability.toFixed(4)}</span>
                </p>
              )}
            </div>
          </div>

          <div className="text-right">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-[#52525B]">
              {metricLabel}
            </div>
            <div className="font-mono-tabular text-2xl font-extrabold text-[#111111]">
              {displayPct.toFixed(1)}%
            </div>
            <div className="font-mono-tabular mt-0.5 inline-flex items-center gap-1 text-[11px] text-[#52525B]">
              <Clock className="size-3" />
              {result.latency_ms} ms{labMode ? ` · ${result.model_mode}` : ""}
            </div>
          </div>
        </div>

        <div className="mt-3.5 h-2 w-full overflow-hidden rounded-full bg-white/80">
          <div
            className={cn("h-full transition-transform duration-200 ease-out", barClass)}
            style={{ width: `${displayPct}%` }}
          />
        </div>
      </div>

      <div className="p-4 sm:p-5">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[#52525B]">
          {copy.whatToDoTitle}
        </h3>
        <ul className="mt-2 space-y-1.5 text-sm text-[#111111]">
          {checklist.map((item, idx) => (
            <li key={idx} className="flex items-start gap-2">
              <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-[#111111]" aria-hidden="true" />
              <span>{item}</span>
            </li>
          ))}
        </ul>

        {showActions && (
          <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-[#E4E4E0] pt-3.5">
            <button
              type="button"
              onClick={onOpenReport}
              className="inline-flex min-h-[44px] items-center gap-1.5 rounded-md bg-[#C81E1E] px-3.5 py-2 text-xs font-semibold text-white transition-colors duration-150 hover:bg-[#991B1B] sm:min-h-[38px]"
            >
              <FileWarning className="size-4" />
              {copy.reportBtn}
            </button>

            <a
              href="tel:1930"
              className="inline-flex min-h-[44px] items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-[#F7F7F5] px-3.5 py-2 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#EAEAE5] sm:min-h-[38px]"
            >
              <PhoneCall className="size-4 text-[#C81E1E]" />
              {copy.call1930Btn}
            </a>

            <button
              type="button"
              onClick={handleWarnFamily}
              className="inline-flex min-h-[44px] items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-white px-3.5 py-2 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#F7F7F5] sm:min-h-[38px]"
            >
              {warnCopied ? <Check className="size-4 text-[#047857]" /> : <Share2 className="size-4" />}
              {warnCopied ? copy.copiedWarning : copy.warnFamilyBtn}
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
