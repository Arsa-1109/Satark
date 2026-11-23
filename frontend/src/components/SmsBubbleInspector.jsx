import { useState } from "react";
import { MessageSquareWarning, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "../utils/cn";

function highlightMessage(text, triggers) {
  if (!triggers || triggers.length === 0) {
    return [{ text, polarity: null, reason: null, triggerIdx: null }];
  }

  const matches = [];
  const lower = text.toLowerCase();
  triggers.forEach((t, triggerIdx) => {
    const idx = lower.indexOf(t.span.toLowerCase());
    if (idx !== -1) {
      matches.push({
        start: idx,
        end: idx + t.span.length,
        polarity: t.polarity,
        reason: t.reason,
        triggerIdx,
      });
    }
  });
  matches.sort((a, b) => a.start - b.start);

  const clean = [];
  let cursor = 0;
  for (const m of matches) {
    if (m.start >= cursor) {
      clean.push(m);
      cursor = m.end;
    }
  }

  const segments = [];
  let pos = 0;
  for (const m of clean) {
    if (m.start > pos) {
      segments.push({ text: text.slice(pos, m.start), polarity: null, reason: null, triggerIdx: null });
    }
    segments.push({
      text: text.slice(m.start, m.end),
      polarity: m.polarity,
      reason: m.reason,
      triggerIdx: m.triggerIdx,
    });
    pos = m.end;
  }
  if (pos < text.length) {
    segments.push({ text: text.slice(pos), polarity: null, reason: null, triggerIdx: null });
  }
  return segments;
}

export function SmsBubbleInspector({ rawText, result }) {
  const [showDiff, setShowDiff] = useState(false);
  const [selectedTriggerIdx, setSelectedTriggerIdx] = useState(null);

  if (!result) return null;

  const triggers = result.forensics?.triggers || [];
  const segments = highlightMessage(rawText || "", triggers);
  const obfs = result.obfuscations_detected || [];

  const toggleTrigger = (idx) => {
    setSelectedTriggerIdx((prev) => (prev === idx ? null : idx));
  };

  return (
    <section className="rounded-xl border border-[#E4E4E0] bg-white p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <MessageSquareWarning className="size-4 text-[#111111]" />
          <h2 className="text-balance text-xs font-bold uppercase tracking-wider text-[#111111]">
            Highlighted Signals in Message
          </h2>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {result.forensics?.sender_analysis?.impersonation_risk && (
            <span className="font-mono-tabular rounded border border-[#C81E1E]/40 bg-[#C81E1E]/10 px-2 py-0.5 text-[11px] font-semibold text-[#C81E1E]">
              SPOOFED SENDER: {result.forensics.sender_analysis.raw}
            </span>
          )}
          {result.forensics?.dlt_header && (
            <span className="font-mono-tabular rounded border border-[#047857]/30 bg-[#047857]/5 px-2 py-0.5 text-[11px] font-semibold text-[#047857]">
              DLT: {result.forensics.dlt_header}
            </span>
          )}
        </div>
      </div>

      <div className="rounded-xl rounded-tl-xs border border-[#E4E4E0] bg-[#F7F7F5] p-4 text-sm leading-relaxed text-[#111111]">
        {segments.map((seg, i) =>
          seg.polarity ? (
            <button
              key={i}
              type="button"
              onClick={() => toggleTrigger(seg.triggerIdx)}
              aria-pressed={selectedTriggerIdx === seg.triggerIdx}
              className={cn(
                "inline rounded px-1 py-0.5 text-left font-medium transition-shadow duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#111111]",
                seg.polarity === "risk"
                  ? "bg-[#C81E1E]/15 text-[#C81E1E] underline decoration-[#C81E1E]/50 decoration-dotted"
                  : "bg-[#047857]/15 text-[#047857]",
                selectedTriggerIdx === seg.triggerIdx && "ring-2 ring-[#111111]"
              )}
            >
              {seg.text}
            </button>
          ) : (
            <span key={i}>{seg.text}</span>
          )
        )}
      </div>

      {triggers.length > 0 && (
        <div className="mt-3">
          <p className="mb-1.5 text-[11px] text-[#52525B]">
            Tap any highlighted phrase or badge to inspect its reason:
          </p>
          <div className="flex flex-wrap gap-1.5">
            {triggers.map((t, i) => {
              const isSelected = selectedTriggerIdx === i;
              return (
                <button
                  key={i}
                  type="button"
                  onClick={() => toggleTrigger(i)}
                  aria-pressed={isSelected}
                  className={cn(
                    "inline-flex min-h-[36px] items-center gap-1 rounded border px-2.5 py-1 text-left text-[11px] font-medium transition-all duration-150 sm:min-h-[28px]",
                    t.polarity === "risk"
                      ? "border-[#C81E1E]/25 bg-[#C81E1E]/5 text-[#C81E1E]"
                      : "border-[#047857]/25 bg-[#047857]/5 text-[#047857]",
                    isSelected && "border-[#111111] ring-1 ring-[#111111]"
                  )}
                >
                  <span className="font-mono-tabular font-semibold">{t.span}</span>
                  <span className="text-[#52525B]">· {t.reason}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {obfs.length > 0 && (
        <div className="mt-3 border-t border-[#E4E4E0] pt-3">
          <button
            type="button"
            onClick={() => setShowDiff((v) => !v)}
            className="flex min-h-[36px] items-center gap-1.5 text-xs font-semibold text-[#B45309]"
          >
            {showDiff ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
            De-obfuscated ({obfs.join(", ")})
          </button>
          {showDiff && (
            <div className="font-mono-tabular mt-2 rounded border border-[#E4E4E0] bg-[#F7F7F5] p-3 text-xs text-[#111111]">
              {result.normalised_text}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
