import { useState } from "react";
import { History, ChevronDown, ChevronUp, Trash2, X } from "lucide-react";
import { cn } from "../utils/cn";

export function HistoryDrawer({ items, onSelectItem, onRemoveItem, onClearAll }) {
  const [open, setOpen] = useState(false);

  if (!items || items.length === 0) return null;

  return (
    <section className="rounded-xl border border-[#E4E4E0] bg-white">
      <div className="flex items-center justify-between px-4 py-3 sm:px-5">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="flex min-h-[40px] flex-1 items-center gap-2 text-left text-xs font-bold uppercase tracking-wider text-[#111111]"
        >
          <History className="size-4 text-[#52525B]" />
          <span>Recent Scans ({items.length})</span>
          <span className="font-normal normal-case text-[#52525B]">· Saved only on this device</span>
          {open ? <ChevronUp className="ml-auto size-4 text-[#52525B]" /> : <ChevronDown className="ml-auto size-4 text-[#52525B]" />}
        </button>

        {open && (
          <button
            type="button"
            onClick={onClearAll}
            className="ml-3 inline-flex min-h-[36px] items-center gap-1 rounded border border-[#E4E4E0] bg-[#F7F7F5] px-2.5 py-1 text-[11px] font-semibold text-[#52525B] hover:text-[#C81E1E]"
          >
            <Trash2 className="size-3.5" />
            Clear All
          </button>
        )}
      </div>

      {open && (
        <ul className="divide-y divide-[#E4E4E0] border-t border-[#E4E4E0]">
          {items.map((item) => {
            const isBorderline = item.certainty_tier?.startsWith("Borderline");
            const isScam = item.verdict === "SCAM";
            const badgeClass = isBorderline
              ? "border-[#B45309]/30 bg-[#B45309]/10 text-[#B45309]"
              : isScam
              ? "border-[#C81E1E]/30 bg-[#C81E1E]/10 text-[#C81E1E]"
              : "border-[#047857]/30 bg-[#047857]/10 text-[#047857]";

            return (
              <li key={item.id} className="flex items-center justify-between gap-2 px-4 py-2.5 sm:px-5">
                <button
                  type="button"
                  onClick={() => onSelectItem(item)}
                  className="min-w-0 flex-1 text-left hover:opacity-80"
                >
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className={cn("rounded border px-1.5 py-0.5 text-[10px] font-bold", badgeClass)}>
                      {isBorderline ? "BORDERLINE" : item.verdict}
                    </span>
                    {item.category && (
                      <span className="text-[11px] font-semibold text-[#111111]">{item.category}</span>
                    )}
                    <span className="font-mono-tabular text-[10px] text-[#52525B]">
                      {new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                  </div>
                  <p className="mt-1 truncate text-xs text-[#52525B]">{item.preview}</p>
                </button>

                <button
                  type="button"
                  onClick={() => onRemoveItem(item.id)}
                  aria-label="Delete scan from history"
                  className="flex size-9 shrink-0 items-center justify-center rounded text-[#52525B] hover:bg-[#F7F7F5] hover:text-[#111111]"
                >
                  <X className="size-4" />
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
