import { Shield, WifiOff, Cpu, ClipboardCheck, Activity, Lock, FlaskConical, Languages } from "lucide-react";
import { cn } from "../utils/cn";

export function TopBar({
  runtimeMode,
  modelMode,
  clipboardSupported,
  clipboardEnabled,
  onToggleClipboard,
  onRunOfflineAudit,
  auditLoading,
  activeTab,
  onTabChange,
  labMode,
  onToggleLabMode,
  lang,
  onToggleLang,
}) {
  const modeLabel =
    runtimeMode === "pwa"
      ? "Offline PWA (JS)"
      : modelMode === "full_onnx_hybrid"
      ? "Full ONNX Hybrid"
      : "Lite Fallback";

  return (
    <header className="sticky top-0 z-20 border-b border-[#E4E4E0] bg-[#F7F7F5]/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-2 px-4 py-2.5 sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-[#111111] text-[#F7F7F5]">
            <Shield className="size-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-bold tracking-tight text-[#111111]">SATARK</span>
              {labMode && (
                <span className="font-mono-tabular rounded border border-[#E4E4E0] bg-white px-1.5 py-0.5 text-[11px] font-semibold text-[#52525B]">
                  TV004
                </span>
              )}
            </div>
            <p className="hidden text-[11px] text-[#52525B] sm:block">
              100% On-Device Indian Scam Shield
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 sm:gap-2">
          <button
            type="button"
            onClick={onToggleLang}
            aria-label="Toggle English or Hinglish guidance"
            className="inline-flex min-h-[44px] items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-white px-2.5 py-1.5 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#EFEFEC] sm:min-h-[34px]"
          >
            <Languages className="size-3.5 text-[#52525B]" />
            <span>{lang === "hi" ? "Hinglish" : "EN"}</span>
          </button>

          {runtimeMode === "laptop" && clipboardSupported && (
            <button
              type="button"
              onClick={onToggleClipboard}
              aria-pressed={clipboardEnabled}
              className={cn(
                "inline-flex min-h-[44px] items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-semibold transition-colors duration-150 sm:min-h-[34px]",
                clipboardEnabled
                  ? "border-[#047857] bg-[#047857]/10 text-[#047857]"
                  : "border-[#E4E4E0] bg-white text-[#52525B] hover:text-[#111111]"
              )}
            >
              <ClipboardCheck className="size-3.5" />
              <span className="hidden sm:inline">
                {clipboardEnabled ? "Guardian: ON" : "Guardian"}
              </span>
              {clipboardEnabled && (
                <span className="size-1.5 rounded-full bg-[#047857]" aria-hidden="true" />
              )}
            </button>
          )}

          <button
            type="button"
            onClick={onToggleLabMode}
            aria-pressed={labMode}
            className={cn(
              "inline-flex min-h-[44px] items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-semibold transition-colors duration-150 sm:min-h-[34px]",
              labMode
                ? "border-[#111111] bg-[#111111] text-white"
                : "border-[#E4E4E0] bg-white text-[#52525B] hover:text-[#111111]"
            )}
          >
            <FlaskConical className="size-3.5" />
            <span>Lab Mode</span>
          </button>
        </div>
      </div>

      {labMode && (
        <div className="border-t border-[#E4E4E0] bg-white/80">
          <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-4 py-2 sm:px-6">
            <div
              role="tablist"
              aria-label="Evaluator views"
              className="flex items-center gap-1 rounded-md border border-[#E4E4E0] bg-[#F7F7F5] p-0.5"
            >
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === "scanner"}
                onClick={() => onTabChange("scanner")}
                className={cn(
                  "min-h-[40px] rounded px-3 py-1 text-xs font-semibold transition-colors duration-150 sm:min-h-[30px]",
                  activeTab === "scanner"
                    ? "bg-[#111111] text-white"
                    : "text-[#52525B] hover:text-[#111111]"
                )}
              >
                Scanner
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === "benchmarks"}
                onClick={() => onTabChange("benchmarks")}
                className={cn(
                  "flex min-h-[40px] items-center gap-1.5 rounded px-3 py-1 text-xs font-semibold transition-colors duration-150 sm:min-h-[30px]",
                  activeTab === "benchmarks"
                    ? "bg-[#111111] text-white"
                    : "text-[#52525B] hover:text-[#111111]"
                )}
              >
                <Activity className="size-3.5" />
                Benchmarks
              </button>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono-tabular inline-flex min-h-[32px] items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-[#F7F7F5] px-2.5 py-1 text-xs font-medium text-[#111111]">
                {runtimeMode === "pwa" ? (
                  <WifiOff className="size-3.5 text-[#047857]" />
                ) : (
                  <Cpu className="size-3.5 text-[#111111]" />
                )}
                {modeLabel}
              </span>

              {runtimeMode === "laptop" && (
                <button
                  type="button"
                  onClick={onRunOfflineAudit}
                  disabled={auditLoading}
                  className="inline-flex min-h-[40px] items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-white px-2.5 py-1 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#EFEFEC] disabled:opacity-50 sm:min-h-[32px]"
                >
                  <Lock className="size-3.5 text-[#047857]" />
                  {auditLoading ? "Auditing..." : "Zero-Network Audit"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
