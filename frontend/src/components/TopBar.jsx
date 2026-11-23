import { Shield, WifiOff, Cpu, ClipboardCheck, Activity, Lock } from "lucide-react";
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
}) {
  const modeLabel =
    runtimeMode === "pwa"
      ? "Offline PWA (JS)"
      : modelMode === "full_onnx_hybrid"
      ? "Full ONNX Hybrid"
      : "Lite Fallback";

  return (
    <header className="sticky top-0 z-20 border-b border-[#E4E4E0] bg-[#F7F7F5]/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-md bg-[#111111] text-[#F7F7F5]">
            <Shield className="size-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-bold tracking-tight text-[#111111]">SATARK</span>
              <span className="font-mono-tabular rounded border border-[#E4E4E0] bg-white px-1.5 py-0.5 text-[11px] font-semibold text-[#52525B]">
                TV004
              </span>
            </div>
            <p className="text-[11px] text-[#52525B]">100% On-Device Indian Scam Detector</p>
          </div>
        </div>

        <nav className="flex items-center gap-1 rounded-md border border-[#E4E4E0] bg-white p-0.5">
          <button
            type="button"
            onClick={() => onTabChange("scanner")}
            className={cn(
              "rounded px-3 py-1.5 text-xs font-semibold transition-colors duration-150",
              activeTab === "scanner" ? "bg-[#111111] text-white" : "text-[#52525B] hover:text-[#111111]"
            )}
          >
            Scanner
          </button>
          <button
            type="button"
            onClick={() => onTabChange("benchmarks")}
            className={cn(
              "flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-semibold transition-colors duration-150",
              activeTab === "benchmarks" ? "bg-[#111111] text-white" : "text-[#52525B] hover:text-[#111111]"
            )}
          >
            <Activity className="size-3.5" />
            Benchmarks
          </button>
        </nav>

        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono-tabular inline-flex items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-white px-2.5 py-1.5 text-xs font-medium text-[#111111]">
            {runtimeMode === "pwa" ? <WifiOff className="size-3.5 text-[#047857]" /> : <Cpu className="size-3.5 text-[#111111]" />}
            {modeLabel}
          </span>

          {runtimeMode === "laptop" && clipboardSupported && (
            <button
              type="button"
              onClick={onToggleClipboard}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-semibold transition-colors duration-150",
                clipboardEnabled
                  ? "border-[#047857] bg-[#047857]/10 text-[#047857]"
                  : "border-[#E4E4E0] bg-white text-[#52525B] hover:text-[#111111]"
              )}
            >
              <ClipboardCheck className="size-3.5" />
              {clipboardEnabled ? "Clipboard Guardian: ON" : "Clipboard Guardian"}
            </button>
          )}

          {runtimeMode === "laptop" && (
            <button
              type="button"
              onClick={onRunOfflineAudit}
              disabled={auditLoading}
              className="inline-flex items-center gap-1.5 rounded-md border border-[#E4E4E0] bg-white px-2.5 py-1.5 text-xs font-semibold text-[#111111] transition-colors duration-150 hover:bg-[#EFEFEC] disabled:opacity-50"
            >
              <Lock className="size-3.5 text-[#047857]" />
              {auditLoading ? "Auditing..." : "Zero-Network Audit"}
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
