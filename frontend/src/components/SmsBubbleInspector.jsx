import { useState } from "react";
import { MessageSquareWarning, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "../utils/cn";

function highlightMessage(text, triggers) {
  if (!triggers || triggers.length === 0) return [{ text, polarity: null, reason: null }];

  const matches = [];
  const lower = text.toLowerCase();
  for (const t of triggers) {
    const idx = lower.indexOf(t.span.toLowerCase());
    if (idx !== -1) {
      matches.push({ start: idx, end: idx + t.span.length, polarity: t.polarity, reason: t.reason });
    }
  }
  matches.sort((a, b) => a.start - b.start);

  // Merge non-overlapping spans
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
    if (m.start > pos) segments.push({ text: text.slice(pos, m.start), polarity: null, reason: null });
    segments.push({ text: text.slice(m.start, m.end), polarity: m.polarity, reason: m.reason });
    pos = m.end;
  }
  if (pos < text.length) segments.push({ text: text.slice(pos), polarity: null, reason: null });
  return segments;
}

export function SmsBubbleInspector({ rawText, result }) {
  const [showDiff, setShowDiff] = useState(false);
  if (!result) return null;

  const triggers = result.forensics?.triggers || [];
  const segments = highlightMessage(rawText, triggers);
  const obfs = result.obfuscations_detected || [];

  return (
    <section className="rounded-lg border border-[#E4E4E0] bg-white p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <MessageSquareWarning className="size-4 text-[#111111]" />
          <h2 className="text-balance text-xs font-bold uppercase tracking-wider text-[#111111]">
            SMS / WhatsApp Bubble Inspector
          </h2>
        </div>
        {result.forensics?.dlt_header && (
          <span className="font-mono-tabular rounded border border-[#047857]/30 bg-[#047857]/5 px-2 py-0.5 text-[11px] font-semibold text-[#047857]">
            DLT: {result.forensics.dlt_header}
          </span>
        )}
      </div>

      <div className="rounded-xl rounded-tl-xs border border-[#E4E4E0] bg-[#F7F7F5] p-4 text-sm leading-relaxed text-[#111111]">
        {segments.map((seg, i) =>
          seg.polarity ? (
            <mark
              key={i}
              title={seg.reason}
              className={cn(
                "rounded px-1 py-0.5 font-medium",
                seg.polarity === "risk"
                  ? "bg-[#C81E1E]/15 text-[#C81E1E] underline decoration-[#C81E1E]/50 decoration-dotted"
                  : "bg-[#047857]/15 text-[#047857]"
              )}
            >
              {seg.text}
            </mark>
          ) : (
            <span key={i}>{seg.text}</span>
          )
        )}
      </div>

      {triggers.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {triggers.map((t, i) => (
            <span
              key={i}
              className={cn(
                "inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[11px] font-medium",
                t.polarity === "risk"
                  ? "border-[#C81E1E]/25 bg-[#C81E1E]/5 text-[#C81E1E]"
                  : "border-[#047857]/25 bg-[#047857]/5 text-[#047857]"
              )}
            >
              <span className="font-mono-tabular font-semibold">{t.span}</span>
              <span className="text-[#52525B]">· {t.reason}</span>
            </span>
          ))}
        </div>
      )}

      {obfs.length > 0 && (
        <div className="mt-3 border-t border-[#E4E4E0] pt-3">
          <button
            type="button"
            onClick={() => setShowDiff((v) => !v)}
            className="flex items-center gap-1.5 text-xs font-semibold text-[#B45309]"
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
