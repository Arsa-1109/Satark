import { ClipboardPaste, ScanSearch, Swords, Eraser } from "lucide-react";
import { JUDGE_PRESETS } from "../data/presets";

export function ScannerInput({
  text,
  onChangeText,
  onScan,
  onMutate,
  onPasteAndScan,
  loading,
  error,
  runtimeMode,
}) {
  return (
    <section className="rounded-lg border border-[#E4E4E0] bg-white p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-balance text-sm font-bold uppercase tracking-wider text-[#111111]">
            Message Inspection Console
          </h1>
          <p className="text-pretty text-xs text-[#52525B]">
            Paste any SMS, WhatsApp forward, or payment link. Processed 100% locally.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onPasteAndScan}
            className="inline-flex items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-[#F7F7F5] px-2.5 py-1.5 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#EAEAE5]"
          >
            <ClipboardPaste className="size-3.5" />
            1-Tap Paste &amp; Scan
          </button>
          {text && (
            <button
              type="button"
              onClick={() => onChangeText("")}
              aria-label="Clear message"
              className="inline-flex items-center gap-1 rounded-md border border-[#E4E4E0] bg-white px-2 py-1.5 text-xs font-medium text-[#52525B] hover:text-[#111111]"
            >
              <Eraser className="size-3.5" />
              Clear
            </button>
          )}
        </div>
      </div>

      <textarea
        rows={4}
        value={text}
        onChange={(e) => onChangeText(e.target.value)}
        placeholder="Paste suspicious SMS or WhatsApp text here (English, Hinglish, Devanagari, upi://pay, etc.)..."
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
            onClick={() => onScan(text)}
            disabled={loading || !text.trim()}
            className="inline-flex items-center gap-2 rounded-md bg-[#111111] px-4 py-2 text-xs font-semibold text-white transition-colors duration-150 hover:bg-[#27272A] disabled:opacity-40"
          >
            <ScanSearch className="size-4" />
            {loading ? "Scanning..." : "Analyze Message"}
          </button>

          {runtimeMode === "laptop" && (
            <button
              type="button"
              onClick={onMutate}
              disabled={loading || !text.trim()}
              className="inline-flex items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-white px-3 py-2 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#F7F7F5] disabled:opacity-40"
            >
              <Swords className="size-3.5 text-[#B45309]" />
              Red-Team Mutate
            </button>
          )}
        </div>

        <span className="font-mono-tabular text-[11px] text-[#52525B]">
          {text.length} / 10,000 chars
        </span>
      </div>

      <div className="mt-4 border-t border-[#E4E4E0] pt-3">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[#52525B]">
          Judge Presets (1-Click)
        </div>
        <div className="flex flex-wrap gap-1.5">
          {JUDGE_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => {
                onChangeText(p.text);
                onScan(p.text);
              }}
              className="inline-flex items-center gap-1.5 rounded border border-[#E4E4E0] bg-[#F7F7F5] px-2.5 py-1 text-xs font-medium text-[#111111] transition-colors duration-150 hover:border-[#111111] hover:bg-white"
            >
              <span>{p.label}</span>
              <span className="font-mono-tabular text-[10px] text-[#52525B]">{p.tag}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
