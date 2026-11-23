import { ShieldAlert, ShieldCheck, AlertTriangle, Clock } from "lucide-react";
import { cn } from "../utils/cn";

export function VerdictHero({ result }) {
  if (!result) return null;

  const isScam = result.verdict === "SCAM";
  const isBorderline = result.certainty_tier.startsWith("Borderline");

  const accentBorder = isBorderline
    ? "border-l-4 border-l-[#B45309]"
    : isScam
    ? "border-l-4 border-l-[#C81E1E]"
    : "border-l-4 border-l-[#047857]";

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

  return (
    <section className={cn("rounded-lg border border-[#E4E4E0] bg-white p-4 sm:p-5", accentBorder)}>
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
              <span className="text-xl font-extrabold tracking-tight text-[#111111]">
                {result.verdict}
              </span>
              {result.category && (
                <span className="rounded border border-[#E4E4E0] bg-[#F7F7F5] px-2 py-0.5 text-xs font-semibold text-[#111111]">
                  {result.category}
                </span>
              )}
              {result.sub_type && (
                <span className="rounded border border-[#C81E1E]/30 bg-[#C81E1E]/5 px-2 py-0.5 text-xs font-semibold text-[#C81E1E]">
                  {result.sub_type}
                </span>
              )}
              {result.truncated && (
                <span className="rounded border border-[#B45309]/40 bg-[#B45309]/10 px-2 py-0.5 text-[11px] font-medium text-[#B45309]">
                  Truncated to 5,000 chars
                </span>
              )}
            </div>
            <p className="mt-1 text-xs font-medium text-[#52525B]">
              Certainty: <span className="text-[#111111]">{result.certainty_tier}</span>
              {" · "}
              <span className="font-mono-tabular">p_scam = {result.scam_probability.toFixed(4)}</span>
            </p>
          </div>
        </div>

        <div className="text-right">
          <div className="font-mono-tabular text-2xl font-extrabold text-[#111111]">
            {result.confidence_pct.toFixed(1)}%
          </div>
          <div className="font-mono-tabular mt-0.5 inline-flex items-center gap-1 text-[11px] text-[#52525B]">
            <Clock className="size-3" />
            {result.latency_ms} ms · {result.model_mode}
          </div>
        </div>
      </div>

      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-[#EFEFEC]">
        <div
          className={cn("h-full transition-transform duration-200 ease-out", barClass)}
          style={{ width: `${Math.min(100, Math.max(0, result.confidence_pct))}%` }}
        />
      </div>
    </section>
  );
}
