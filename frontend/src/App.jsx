import { useEffect, useState, useCallback, useRef } from "react";
import { Bell, ClipboardCheck, PauseCircle } from "lucide-react";
import { TopBar } from "./components/TopBar";
import { ScannerInput } from "./components/ScannerInput";
import { VerdictHero } from "./components/VerdictHero";
import { SmsBubbleInspector } from "./components/SmsBubbleInspector";
import { ForensicDrawers } from "./components/ForensicDrawers";
import { BenchmarkView } from "./components/BenchmarkView";
import { HistoryDrawer } from "./components/HistoryDrawer";
import { PwaInstallBanner } from "./components/PwaInstallBanner";
import { analyzeLocal } from "./engine/localEngine";
import { JUDGE_PRESETS } from "./data/presets";
import { loadHistory, addHistoryEntry, removeHistoryEntry, clearHistory } from "./utils/history";
import { cn } from "./utils/cn";

const DEFAULT_TITLE = "Satark — On-Device Scam Message Detector";

export function App() {
  const [runtimeMode, setRuntimeMode] = useState("pwa"); // "laptop" | "pwa"
  const [modelMode, setModelMode] = useState("pwa_js");
  const [clipboardSupported, setClipboardSupported] = useState(false);
  const [clipboardEnabled, setClipboardEnabled] = useState(false);
  const [clipboardOnlyScam, setClipboardOnlyScam] = useState(false);

  const [labMode, setLabMode] = useState(false);
  const [lang, setLang] = useState("en");
  const [activeTab, setActiveTab] = useState("scanner");

  const [text, setText] = useState("");
  const [sender, setSender] = useState("");
  const [result, setResult] = useState(null);
  const [lastScannedText, setLastScannedText] = useState("");
  const [mutationData, setMutationData] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [auditProof, setAuditProof] = useState(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [collapsedForShare, setCollapsedForShare] = useState(false);
  const [clipboardBanner, setClipboardBanner] = useState(false);
  const [notifPermission, setNotifPermission] = useState(() =>
    typeof Notification !== "undefined" ? Notification.permission : "unsupported"
  );
  const [undoSnapshot, setUndoSnapshot] = useState(null);
  const [history, setHistory] = useState(() => loadHistory());

  const [requestedDrawerTab, setRequestedDrawerTab] = useState(null);
  const [requestedDrawerNonce, setRequestedDrawerNonce] = useState(0);

  const verdictRef = useRef(null);
  const drawerRef = useRef(null);
  const undoTimerRef = useRef(null);

  const recordScan = useCallback((rawInput, activeSender, scanResult, source) => {
    setResult(scanResult);
    setLastScannedText(rawInput);
    setHistory(addHistoryEntry({ rawText: rawInput, sender: activeSender, result: scanResult, source }));
  }, []);

  const runScan = useCallback(
    async (inputText, source = "text", modeOverride = null, senderOverride = undefined, shouldScroll = true) => {
      const mode = modeOverride || runtimeMode;
      const activeSender = senderOverride !== undefined ? senderOverride : sender;
      const senderVal = activeSender && activeSender.trim() ? activeSender.trim() : null;
      if (!inputText || !inputText.trim()) {
        setError("Please enter or paste a message to scan.");
        return;
      }
      setError(null);
      setLoading(true);
      setMutationData(null);
      setClipboardBanner(false);
      try {
        let scanRes;
        if (mode === "laptop") {
          const resp = await fetch("./api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: inputText, source, sender: senderVal }),
          });
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${resp.status})`);
          }
          scanRes = await resp.json();
        } else {
          scanRes = analyzeLocal(inputText, source, null, senderVal);
        }
        recordScan(inputText, senderVal || "", scanRes, source);
        if (shouldScroll) {
          setTimeout(() => {
            verdictRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }, 50);
        }
      } catch (e) {
        try {
          const fallbackRes = analyzeLocal(inputText, source, null, senderVal);
          setRuntimeMode("pwa");
          setModelMode("pwa_js");
          recordScan(inputText, senderVal || "", fallbackRes, source);
        } catch (localErr) {
          setError(localErr.message || e.message);
        }
      } finally {
        setLoading(false);
      }
    },
    [runtimeMode, sender, recordScan]
  );

  // Startup probe (/api/health with 400ms timeout) + Android share_target params
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sharedText = [params.get("text"), params.get("url")].filter(Boolean).join(" ").trim();
    const sharedTitle = (params.get("title") || "").trim();
    const hasShare = Boolean(sharedText || sharedTitle);
    const initialText = sharedText || sharedTitle || "";
    const initialSender = sharedText && sharedTitle ? sharedTitle : "";

    if (hasShare) {
      window.history.replaceState({}, "", window.location.pathname);
      setText(initialText);
      if (initialSender) setSender(initialSender);
      setCollapsedForShare(true);
    }

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 400);

    fetch("./api/health", { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((health) => {
        clearTimeout(timer);
        setRuntimeMode("laptop");
        setModelMode(health.model_mode);
        setClipboardSupported(Boolean(health.clipboard_supported));
        setClipboardEnabled(Boolean(health.clipboard_enabled));
        setClipboardOnlyScam(Boolean(health.clipboard_only_alert_scam));
        if (hasShare && initialText) {
          runScan(initialText, "share_target", "laptop", initialSender, true);
        }
        fetch("./api/benchmarks").then((r) => r.json()).then(setBenchmarks).catch(() => {});
      })
      .catch(() => {
        clearTimeout(timer);
        setRuntimeMode("pwa");
        setModelMode("pwa_js");
        if (hasShare && initialText) {
          runScan(initialText, "share_target", "pwa", initialSender, true);
        }
        fetch("./benchmarks_summary.json").then((r) => r.json()).then(setBenchmarks).catch(() => {});
      });
  }, []);

  // Restore document.title when returning to tab
  useEffect(() => {
    const onVis = () => {
      if (!document.hidden) {
        document.title = DEFAULT_TITLE;
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  // SSE Clipboard Guardian subscription in Laptop mode
  useEffect(() => {
    if (runtimeMode !== "laptop") return;
    const es = new EventSource("./api/clipboard/stream");
    es.addEventListener("scan", (ev) => {
      try {
        const data = JSON.parse(ev.data);
        const rawInput = data.raw_text || data.normalised_text || "";
        setText(rawInput);
        setCollapsedForShare(false);
        setMutationData(null);
        setActiveTab("scanner");
        setClipboardBanner(true);
        recordScan(rawInput, "", data, "clipboard");

        const isAlert = data.verdict === "SCAM" || data.certainty_tier?.startsWith("Borderline");
        if (document.hidden && isAlert) {
          document.title = "[SCAM!] Satark";
          if (typeof Notification !== "undefined" && Notification.permission === "granted") {
            new Notification("Satark: Suspicious Message Copied", {
              body: `${data.verdict}${data.category ? ` (${data.category})` : ""} — ${rawInput.slice(0, 90)}`,
              icon: "./icon-192.png",
            });
          }
        }
      } catch {
        // ignore malformed event
      }
    });
    return () => es.close();
  }, [runtimeMode, recordScan]);

  const updateClipboardConfig = async (patch) => {
    try {
      const resp = await fetch("./api/clipboard/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (resp.ok) {
        const data = await resp.json();
        setClipboardEnabled(Boolean(data.enabled));
        setClipboardOnlyScam(Boolean(data.only_alert_scam));
      }
    } catch {
      // ignore
    }
  };

  const handleToggleClipboard = () => {
    updateClipboardConfig({ enabled: !clipboardEnabled });
  };

  const handleToggleOnlyScam = () => {
    updateClipboardConfig({ only_alert_scam: !clipboardOnlyScam });
  };

  const handleEnableDesktopAlerts = async () => {
    if (typeof Notification === "undefined") return;
    const perm = await Notification.requestPermission();
    setNotifPermission(perm);
  };

  const handleMutate = async () => {
    if (!text.trim() || runtimeMode !== "laptop") return;
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("./api/mutate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, level: "medium" }),
      });
      if (!resp.ok) throw new Error(`Mutate failed (${resp.status})`);
      const data = await resp.json();
      setResult(data.original);
      setLastScannedText(text);
      setMutationData(data);
      setRequestedDrawerTab("mutate");
      setRequestedDrawerNonce((n) => n + 1);
      setTimeout(() => {
        drawerRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 50);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePasteAndScan = async () => {
    try {
      const clip = await navigator.clipboard.readText();
      if (clip && clip.trim()) {
        setText(clip);
        setCollapsedForShare(false);
        runScan(clip, "clipboard");
      } else {
        setError("Clipboard is empty — copy a message first, then tap Paste & Scan.");
        setTimeout(() => setError((prev) => (prev && /clipboard is empty/i.test(prev) ? null : prev)), 4000);
      }
    } catch {
      const isTouch = typeof window !== "undefined" && ("ontouchstart" in window || navigator.maxTouchPoints > 0);
      setError(
        isTouch
          ? "Clipboard permission blocked. Tap and hold the message box below to Paste."
          : "Clipboard permission denied. Click inside the box and press Ctrl+V (or Cmd+V)."
      );
    }
  };

  const handleClearAll = () => {
    if (undoTimerRef.current) clearTimeout(undoTimerRef.current);
    setUndoSnapshot({ text, sender, result, lastScannedText, mutationData });
    setText("");
    setSender("");
    setResult(null);
    setLastScannedText("");
    setMutationData(null);
    setClipboardBanner(false);
    setError(null);
    undoTimerRef.current = setTimeout(() => setUndoSnapshot(null), 5000);
  };

  const handleUndoClear = () => {
    if (!undoSnapshot) return;
    if (undoTimerRef.current) clearTimeout(undoTimerRef.current);
    setText(undoSnapshot.text);
    setSender(undoSnapshot.sender);
    setResult(undoSnapshot.result);
    setLastScannedText(undoSnapshot.lastScannedText);
    setMutationData(undoSnapshot.mutationData);
    setUndoSnapshot(null);
  };

  const handleRunOfflineAudit = async () => {
    setAuditLoading(true);
    try {
      const resp = await fetch("./api/audit/offline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text || JUDGE_PRESETS[0].text }),
      });
      if (resp.ok) {
        setAuditProof(await resp.json());
        setLabMode(true);
        setActiveTab("benchmarks");
      }
    } finally {
      setAuditLoading(false);
    }
  };

  const handleOpenReport = () => {
    setRequestedDrawerTab("report");
    setRequestedDrawerNonce((n) => n + 1);
    setTimeout(() => {
      drawerRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 50);
  };

  const handleSelectHistoryItem = (item) => {
    setText(item.rawText);
    setSender(item.sender || "");
    setCollapsedForShare(false);
    setActiveTab("scanner");
    runScan(item.rawText, item.source || "text", null, item.sender || "", true);
  };

  const handleToggleLabMode = () => {
    setLabMode((prev) => {
      const next = !prev;
      if (!next) setActiveTab("scanner");
      return next;
    });
  };

  const isStale = Boolean(result && text.trim() && text !== lastScannedText);

  return (
    <div className="min-h-dvh bg-[#F7F7F5] text-[#111111]">
      <TopBar
        runtimeMode={runtimeMode}
        modelMode={modelMode}
        clipboardSupported={clipboardSupported}
        clipboardEnabled={clipboardEnabled}
        onToggleClipboard={handleToggleClipboard}
        onRunOfflineAudit={handleRunOfflineAudit}
        auditLoading={auditLoading}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        labMode={labMode}
        onToggleLabMode={handleToggleLabMode}
        lang={lang}
        onToggleLang={() => setLang((l) => (l === "en" ? "hi" : "en"))}
      />

      <main className="mx-auto max-w-5xl space-y-3.5 px-3 py-3 sm:space-y-4 sm:px-6 sm:py-5">
        <PwaInstallBanner />

        {clipboardBanner && (
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-[#047857]/30 bg-[#047857]/[0.06] px-4 py-2.5 text-xs">
            <div className="flex items-center gap-2 font-medium text-[#111111]">
              <ClipboardCheck className="size-4 text-[#047857]" />
              <span>Scanned automatically from Clipboard</span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleToggleOnlyScam}
                aria-pressed={clipboardOnlyScam}
                className={cn(
                  "rounded border px-2.5 py-1 text-[11px] font-semibold",
                  clipboardOnlyScam
                    ? "border-[#111111] bg-[#111111] text-white"
                    : "border-[#E4E4E0] bg-white text-[#111111]"
                )}
              >
                {clipboardOnlyScam ? "Alert: Scams Only" : "Alert: All Copies"}
              </button>
              {notifPermission === "default" && (
                <button
                  type="button"
                  onClick={handleEnableDesktopAlerts}
                  className="inline-flex items-center gap-1 rounded border border-[#E4E4E0] bg-white px-2.5 py-1 text-[11px] font-semibold text-[#111111]"
                >
                  <Bell className="size-3" />
                  Enable Desktop Alerts
                </button>
              )}
              <button
                type="button"
                onClick={handleToggleClipboard}
                className="inline-flex items-center gap-1 rounded border border-[#E4E4E0] bg-white px-2.5 py-1 text-[11px] font-semibold text-[#52525B] hover:text-[#111111]"
              >
                <PauseCircle className="size-3" />
                Pause Guardian
              </button>
            </div>
          </div>
        )}

        {labMode && activeTab === "benchmarks" ? (
          <BenchmarkView benchmarks={benchmarks} auditProof={auditProof} />
        ) : (
          <>
            {collapsedForShare ? (
              <>
                <VerdictHero
                  result={result}
                  verdictRef={verdictRef}
                  lang={lang}
                  labMode={labMode}
                  onOpenReport={handleOpenReport}
                />
                <ScannerInput
                  text={text}
                  onChangeText={setText}
                  sender={sender}
                  onChangeSender={setSender}
                  onScan={(t, src) => runScan(t, src || "text")}
                  onMutate={handleMutate}
                  onPasteAndScan={handlePasteAndScan}
                  onClearAll={handleClearAll}
                  onUndoClear={handleUndoClear}
                  hasUndo={Boolean(undoSnapshot)}
                  isStale={isStale}
                  loading={loading}
                  error={error}
                  runtimeMode={runtimeMode}
                  labMode={labMode}
                  lang={lang}
                  collapsedForShare={true}
                  onExpandFromShare={() => setCollapsedForShare(false)}
                />
              </>
            ) : (
              <>
                <ScannerInput
                  text={text}
                  onChangeText={setText}
                  sender={sender}
                  onChangeSender={setSender}
                  onScan={(t, src) => runScan(t, src || "text")}
                  onMutate={handleMutate}
                  onPasteAndScan={handlePasteAndScan}
                  onClearAll={handleClearAll}
                  onUndoClear={handleUndoClear}
                  hasUndo={Boolean(undoSnapshot)}
                  isStale={isStale}
                  loading={loading}
                  error={error}
                  runtimeMode={runtimeMode}
                  labMode={labMode}
                  lang={lang}
                  collapsedForShare={false}
                  onExpandFromShare={() => setCollapsedForShare(false)}
                />
                <VerdictHero
                  result={result}
                  verdictRef={verdictRef}
                  lang={lang}
                  labMode={labMode}
                  onOpenReport={handleOpenReport}
                />
              </>
            )}

            <SmsBubbleInspector rawText={text} result={result} />
            <ForensicDrawers
              result={result}
              mutationData={mutationData}
              requestedTab={requestedDrawerTab}
              requestedTabNonce={requestedDrawerNonce}
              drawerRef={drawerRef}
            />
            <HistoryDrawer
              items={history}
              onSelectItem={handleSelectHistoryItem}
              onRemoveItem={(id) => setHistory(removeHistoryEntry(id))}
              onClearAll={() => setHistory(clearHistory())}
            />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
