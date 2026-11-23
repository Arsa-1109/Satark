import { useEffect, useState } from "react";
import { ClipboardPaste, ScanSearch, Swords, Eraser, Undo2, Plus, ChevronDown, Share2 } from "lucide-react";
import { JUDGE_PRESETS } from "../data/presets";
import { UI_COPY } from "../data/guidance";

const INLINE_DLT_RE = /^\s*([A-Z]{2}-[A-Z0-9]{3,12}(?:-[STPG])?)\s*:/i;

const CONSUMER_EXAMPLES = [
  { preset: JUDGE_PRESETS[0], label: "Fake SBI KYC" },
  { preset: JUDGE_PRESETS[1], label: "UPI Cashback Trap" },
  { preset: JUDGE_PRESETS[2], label: "Digital Arrest Call" },
  { preset: JUDGE_PRESETS[4], label: "Genuine Bank SMS" },
];

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
          className="inline-flex min-h-[44px] items-center gap-1.5 rounded-lg border border-[#E4E4E0] bg-[#F7F7F5] px-3 py-1.5 text-xs font-semibold text-[#111111] hover:bg-[#EAEAE5] sm:min-h-[34px]"
        >
          <ChevronDown className="size-3.5" />
          Scan another message
        </button>
      </section>
    );
  }

  const hasText = Boolean(text && text.trim());
  const isSoftNotice = Boolean(error && /clipboard is empty/i.test(error));

  return (
    <section className="rounded-xl border border-[#E4E4E0] bg-white p-3.5 sm:p-5">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <h1 className="text-balance text-[15px] font-bold tracking-tight text-[#111111] sm:text-base">
            {labMode ? "Message Inspection Console" : copy.emptyTitle}
          </h1>
          <p className="text-pretty mt-0.5 text-xs text-[#52525B]">
            {copy.emptySubtitle}
          </p>
        </div>

        {hasUndo && (
          <button
            type="button"
            onClick={onUndoClear}
            className="inline-flex min-h-[38px] shrink-0 items-center gap-1 rounded-lg border border-[#111111] bg-[#111111] px-3 py-1.5 text-xs font-semibold text-white sm:min-h-[34px]"
          >
            <Undo2 className="size-3.5" />
            Undo
          </button>
        )}
      </div>

      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
        <label htmlFor="satark-message-input" className="text-xs font-semibold text-[#52525B]">
          Message text
        </label>

        <div className="flex items-center gap-2">
          {detectedInlineSender && (
            <span className="font-mono-tabular rounded border border-[#047857]/30 bg-[#047857]/5 px-2 py-0.5 text-[11px] font-medium text-[#047857]">
              Sender: {detectedInlineSender}
            </span>
          )}
          {!showSender && (
            <button
              type="button"
              onClick={() => setShowSender(true)}
              className="inline-flex min-h-[32px] items-center gap-1 text-xs font-medium text-[#52525B] hover:text-[#111111]"
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
            className="font-mono-tabular min-h-[44px] w-full rounded-lg border border-[#E4E4E0] bg-[#F7F7F5] px-3 py-1.5 text-sm text-[#111111] placeholder-[#71717A] focus:border-[#111111] focus:bg-white focus:outline-none sm:min-h-[36px] sm:text-xs"
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
        placeholder="Paste suspicious SMS, WhatsApp forward, or upi://pay link..."
        className="w-full resize-none rounded-lg border border-[#E4E4E0] bg-[#F7F7F5] p-3 text-base leading-relaxed text-[#111111] placeholder-[#71717A] focus:border-[#111111] focus:bg-white focus:outline-none sm:resize-y sm:text-sm"
      />

      {error && (
        <p
          role="alert"
          className={
            isSoftNotice
              ? "mt-2 rounded-lg border border-[#B45309]/25 bg-[#B45309]/[0.06] px-3 py-2 text-xs font-medium text-[#B45309]"
              : "mt-2 rounded-lg border border-[#C81E1E]/25 bg-[#C81E1E]/[0.06] px-3 py-2 text-xs font-medium text-[#C81E1E]"
          }
        >
          {error}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        {!hasText ? (
          <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
            <button
              type="button"
              onClick={onPasteAndScan}
              className="inline-flex min-h-[46px] w-full items-center justify-center gap-2 rounded-lg bg-[#111111] px-4 py-2 text-sm font-semibold text-white transition-colors duration-150 hover:bg-[#27272A] sm:min-h-[38px] sm:w-auto sm:text-xs"
            >
              <ClipboardPaste className="size-4" />
              1-Tap Paste &amp; Scan
            </button>
          </div>
        ) : (
          <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
            <button
              type="button"
              onClick={() => onScan(text, "text")}
              disabled={loading}
              className="inline-flex min-h-[46px] flex-1 items-center justify-center gap-2 rounded-lg bg-[#111111] px-4 py-2 text-sm font-semibold text-white transition-colors duration-150 hover:bg-[#27272A] disabled:opacity-40 sm:min-h-[38px] sm:flex-none sm:text-xs"
            >
              <ScanSearch className="size-4" />
              {loading ? "Scanning..." : "Analyze Message"}
            </button>

            <button
              type="button"
              onClick={onPasteAndScan}
              aria-label="Paste from clipboard and scan"
              className="inline-flex min-h-[46px] items-center justify-center gap-1.5 rounded-lg border border-[#E4E4E0] bg-[#F7F7F5] px-3 py-2 text-xs font-semibold text-[#111111] hover:bg-[#EAEAE5] sm:min-h-[38px]"
            >
              <ClipboardPaste className="size-3.5" />
              <span>Paste</span>
            </button>

            <button
              type="button"
              onClick={onClearAll}
              aria-label="Clear message and result"
              className="inline-flex min-h-[46px] items-center justify-center gap-1 rounded-lg border border-[#E4E4E0] bg-white px-3 py-2 text-xs font-medium text-[#52525B] hover:text-[#111111] sm:min-h-[38px]"
            >
              <Eraser className="size-3.5" />
              <span>Clear</span>
            </button>

            {labMode && runtimeMode === "laptop" && (
              <button
                type="button"
                onClick={onMutate}
                disabled={loading}
                className="inline-flex min-h-[46px] items-center justify-center gap-1.5 rounded-lg border border-[#E4E4E0] bg-white px-3 py-2 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#F7F7F5] disabled:opacity-40 sm:min-h-[38px]"
              >
                <Swords className="size-3.5 text-[#B45309]" />
                Red-Team Mutate
              </button>
            )}
          </div>
        )}

        {text.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            {isStale && (
              <span className="rounded border border-[#B45309]/30 bg-[#B45309]/10 px-2 py-0.5 text-[11px] font-medium text-[#B45309]">
                Edited — tap Analyze to refresh
              </span>
            )}
            <span className="font-mono-tabular text-[11px] text-[#52525B]">
              {text.length} / 10,000
            </span>
          </div>
        )}
      </div>

      <div className="mt-4 border-t border-[#E4E4E0] pt-3">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[#52525B]">
          {labMode ? "Judge Presets (1-Click)" : copy.exampleHeading}
        </div>
        {labMode ? (
          <div className="flex flex-wrap gap-1.5">
            {JUDGE_PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => {
                  onChangeText(p.text);
                  onScan(p.text, "text");
                }}
                className="inline-flex min-h-[40px] items-center gap-1.5 rounded-lg border border-[#E4E4E0] bg-[#F7F7F5] px-2.5 py-1 text-xs font-medium text-[#111111] transition-colors duration-150 hover:border-[#111111] hover:bg-white sm:min-h-[32px]"
              >
                <span>{p.label}</span>
                <span className="font-mono-tabular text-[10px] text-[#52525B]">{p.tag}</span>
              </button>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap">
            {CONSUMER_EXAMPLES.map(({ preset, label }) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => {
                  onChangeText(preset.text);
                  onScan(preset.text, "text");
                }}
                className="inline-flex min-h-[42px] items-center justify-center rounded-lg border border-[#E4E4E0] bg-[#F7F7F5] px-2.5 py-1.5 text-center text-xs font-medium text-[#111111] transition-colors duration-150 hover:border-[#111111] hover:bg-white sm:min-h-[32px] sm:justify-start"
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
