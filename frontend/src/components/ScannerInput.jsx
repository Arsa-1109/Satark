import { useEffect, useState } from "react";
import { ClipboardPaste, ScanSearch, Swords, Eraser, Undo2, Plus, ChevronDown, Share2 } from "lucide-react";
import { JUDGE_PRESETS } from "../data/presets";
import { UI_COPY } from "../data/guidance";

const INLINE_DLT_RE = /^\s*([A-Z]{2}-[A-Z0-9]{3,12}(?:-[STPG])?)\s*:/i;

export function ScannerInput({
  text,
  onChangeText,
  sender,
  onChangeSender,
  onScan,
  onMutate,
  onPasteAndScan,
  onClearAll,
  onUndoClear,
  hasUndo,
  isStale,
  loading,
  error,
  runtimeMode,
  labMode,
  lang,
  collapsedForShare,
  onExpandFromShare,
}) {
  const [showSender, setShowSender] = useState(Boolean(sender));
  const copy = UI_COPY[lang] || UI_COPY.en;

  useEffect(() => {
    if (sender && sender.trim()) {
      setShowSender(true);
    }
  }, [sender]);

  const detectedInlineSender = (() => {
    if (sender && sender.trim()) return null;
    const m = (text || "").match(INLINE_DLT_RE);
    return m ? m[1].toUpperCase() : null;
  })();

  const handlePaste = (e) => {
    const pasted = e.clipboardData?.getData("text");
    if (pasted && pasted.trim()) {
      // Let React update textarea, then immediately scan the pasted message
      setTimeout(() => {
        onScan(pasted, "clipboard");
      }, 0);
    }
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (text.trim() && !loading) {
        onScan(text, "text");
      }
    }
  };

  if (collapsedForShare) {
    return (
      <section className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-[#E4E4E0] bg-white px-4 py-3">
        <div className="flex items-center gap-2 text-xs font-medium text-[#111111]">
          <Share2 className="size-4 text-[#047857]" />
          <span>Shared from another app</span>
        </div>
        <button
          type="button"
          onClick={onExpandFromShare}
          className="inline-flex min-h-[44px] items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-[#F7F7F5] px-3 py-1.5 text-xs font-semibold text-[#111111] hover:bg-[#EAEAE5] sm:min-h-[34px]"
        >
          <ChevronDown className="size-3.5" />
          Scan another message
        </button>
      </section>
    );
  }

  const examplesToShow = labMode ? JUDGE_PRESETS : JUDGE_PRESETS.slice(0, 3);

  return (
    <section className="rounded-xl border border-[#E4E4E0] bg-white p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-balance text-sm font-bold tracking-tight text-[#111111] sm:text-base">
            {labMode ? "Message Inspection Console" : copy.emptyTitle}
          </h1>
          <p className="text-pretty mt-0.5 text-xs text-[#52525B]">
            {copy.emptySubtitle}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onPasteAndScan}
            className="inline-flex min-h-[44px] items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-[#F7F7F5] px-3 py-1.5 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#EAEAE5] sm:min-h-[34px]"
          >
            <ClipboardPaste className="size-3.5" />
            1-Tap Paste &amp; Scan
          </button>

          {(text || sender) && (
            <button
              type="button"
              onClick={onClearAll}
              aria-label="Clear message and result"
              className="inline-flex min-h-[44px] items-center gap-1 rounded-md border border-[#E4E4E0] bg-white px-3 py-1.5 text-xs font-medium text-[#52525B] hover:text-[#111111] sm:min-h-[34px]"
            >
              <Eraser className="size-3.5" />
              Clear
            </button>
          )}

          {hasUndo && (
            <button
              type="button"
              onClick={onUndoClear}
              className="inline-flex min-h-[44px] items-center gap-1 rounded-md border border-[#111111] bg-[#111111] px-3 py-1.5 text-xs font-semibold text-white sm:min-h-[34px]"
            >
              <Undo2 className="size-3.5" />
              Undo
            </button>
          )}
        </div>
      </div>

      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <label htmlFor="satark-message-input" className="text-xs font-semibold text-[#52525B]">
          Suspicious message text
        </label>

        <div className="flex items-center gap-2">
          {detectedInlineSender && (
            <span className="font-mono-tabular rounded border border-[#047857]/30 bg-[#047857]/5 px-2 py-0.5 text-[11px] font-medium text-[#047857]">
              Detected Sender: {detectedInlineSender}
            </span>
          )}
          {!showSender && (
            <button
              type="button"
              onClick={() => setShowSender(true)}
              className="inline-flex min-h-[36px] items-center gap-1 text-xs font-medium text-[#52525B] hover:text-[#111111]"
            >
              <Plus className="size-3.5" />
              Add Sender ID
            </button>
          )}
        </div>
      </div>

      {showSender && (
        <div className="mb-2.5">
          <label htmlFor="satark-sender-input" className="sr-only">
            Optional SMS Sender ID
          </label>
          <input
            id="satark-sender-input"
            type="text"
            value={sender || ""}
            onChange={(e) => onChangeSender(e.target.value)}
            maxLength={64}
            placeholder="Optional Sender ID (e.g. VM-HDFCBK-S or +919876543210)"
            className="font-mono-tabular min-h-[44px] w-full rounded-md border border-[#E4E4E0] bg-[#F7F7F5] px-3 py-1.5 text-xs text-[#111111] placeholder-[#71717A] focus:border-[#111111] focus:bg-white focus:outline-none sm:min-h-[36px]"
          />
        </div>
      )}

      <textarea
        id="satark-message-input"
        rows={4}
        value={text}
        onChange={(e) => onChangeText(e.target.value)}
        onPaste={handlePaste}
        onKeyDown={handleKeyDown}
        placeholder="Paste suspicious SMS or WhatsApp text here (English, Hinglish, Devanagari, upi://pay)..."
        className="w-full resize-y rounded-md border border-[#E4E4E0] bg-[#F7F7F5] p-3 text-sm leading-relaxed text-[#111111] placeholder-[#71717A] focus:border-[#111111] focus:bg-white focus:outline-none"
      />

      {error && (
        <p className="mt-2 text-xs font-medium text-[#C81E1E]" role="alert">
          {error}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => onScan(text, "text")}
            disabled={loading || !text.trim()}
            className="inline-flex min-h-[44px] items-center gap-2 rounded-md bg-[#111111] px-4 py-2 text-xs font-semibold text-white transition-colors duration-150 hover:bg-[#27272A] disabled:opacity-40 sm:min-h-[38px]"
          >
            <ScanSearch className="size-4" />
            {loading ? "Scanning..." : "Analyze Message"}
          </button>

          {labMode && runtimeMode === "laptop" && (
            <button
              type="button"
              onClick={onMutate}
              disabled={loading || !text.trim()}
              className="inline-flex min-h-[44px] items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-white px-3 py-2 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#F7F7F5] disabled:opacity-40 sm:min-h-[38px]"
            >
              <Swords className="size-3.5 text-[#B45309]" />
              Red-Team Mutate
            </button>
          )}

          {isStale && (
            <span className="rounded border border-[#B45309]/30 bg-[#B45309]/10 px-2 py-1 text-[11px] font-medium text-[#B45309]">
              Edited — click Analyze or Ctrl+Enter to refresh
            </span>
          )}
        </div>

        <span className="font-mono-tabular text-[11px] text-[#52525B]">
          {text.length} / 10,000 chars
        </span>
      </div>

      <div className="mt-4 border-t border-[#E4E4E0] pt-3">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[#52525B]">
          {labMode ? "Judge Presets (1-Click)" : copy.exampleHeading}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {examplesToShow.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => {
                onChangeText(p.text);
                onScan(p.text, "text");
              }}
              className="inline-flex min-h-[40px] items-center gap-1.5 rounded border border-[#E4E4E0] bg-[#F7F7F5] px-2.5 py-1 text-xs font-medium text-[#111111] transition-colors duration-150 hover:border-[#111111] hover:bg-white sm:min-h-[32px]"
            >
              <span>{p.label}</span>
              {labMode && (
                <span className="font-mono-tabular text-[10px] text-[#52525B]">{p.tag}</span>
              )}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
