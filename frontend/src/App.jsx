import { useEffect, useState, useCallback } from "react";
import { TopBar } from "./components/TopBar";
import { ScannerInput } from "./components/ScannerInput";
import { VerdictHero } from "./components/VerdictHero";
import { SmsBubbleInspector } from "./components/SmsBubbleInspector";
import { ForensicDrawers } from "./components/ForensicDrawers";
import { BenchmarkView } from "./components/BenchmarkView";
import { analyzeLocal } from "./engine/localEngine";
import { JUDGE_PRESETS } from "./data/presets";

export function App() {
  const [runtimeMode, setRuntimeMode] = useState("pwa"); // "laptop" | "pwa"
  const [modelMode, setModelMode] = useState("pwa_js");
  const [clipboardSupported, setClipboardSupported] = useState(false);
  const [clipboardEnabled, setClipboardEnabled] = useState(false);

  const [activeTab, setActiveTab] = useState("scanner");
  const [text, setText] = useState(JUDGE_PRESETS[0].text);
  const [sender, setSender] = useState("");
  const [result, setResult] = useState(null);
  const [mutationData, setMutationData] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [auditProof, setAuditProof] = useState(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runScan = useCallback(
    async (inputText, source = "text", modeOverride = null, senderOverride = undefined) => {
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
      try {
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
          setResult(await resp.json());
        } else {
          setResult(analyzeLocal(inputText, source, null, senderVal));
        }
      } catch (e) {
        // Auto-fallback to local PWA JS engine if backend drops
        try {
          setResult(analyzeLocal(inputText, source, null, senderVal));
          setRuntimeMode("pwa");
          setModelMode("pwa_js");
        } catch (localErr) {
          setError(localErr.message || e.message);
        }
      } finally {
        setLoading(false);
      }
    },
    [runtimeMode, sender]
  );

  // Startup probe (/api/health with 400ms timeout) + Android share_target params
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sharedText = [params.get("text"), params.get("url")].filter(Boolean).join(" ").trim();
    const sharedTitle = (params.get("title") || "").trim();
    const initialText = sharedText || sharedTitle || JUDGE_PRESETS[0].text;
    const initialSender = sharedText && sharedTitle ? sharedTitle : "";
    const initialSource = sharedText || sharedTitle ? "share_target" : "text";
    setText(initialText);
    if (initialSender) setSender(initialSender);

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
        runScan(initialText, initialSource, "laptop", initialSender);
        fetch("./api/benchmarks").then((r) => r.json()).then(setBenchmarks).catch(() => {});
      })
      .catch(() => {
        clearTimeout(timer);
        setRuntimeMode("pwa");
        setModelMode("pwa_js");
        runScan(initialText, initialSource, "pwa", initialSender);
        fetch("./benchmarks_summary.json").then((r) => r.json()).then(setBenchmarks).catch(() => {});
      });
  }, []);

  // SSE Clipboard Guardian subscription in Laptop mode
  useEffect(() => {
    if (runtimeMode !== "laptop") return;
    const es = new EventSource("./api/clipboard/stream");
    es.addEventListener("scan", (ev) => {
      try {
        const data = JSON.parse(ev.data);
        setText(data.normalised_text || "");
        setResult(data);
        setMutationData(null);
        setActiveTab("scanner");
      } catch {
        // ignore malformed event
      }
    });
    return () => es.close();
  }, [runtimeMode]);

  const handleToggleClipboard = async () => {
    try {
      const resp = await fetch("./api/clipboard/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !clipboardEnabled }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setClipboardEnabled(Boolean(data.enabled));
      }
    } catch {
      // ignore
    }
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
      setMutationData(data);
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
        runScan(clip, "clipboard");
      } else {
        setError("Clipboard is empty.");
      }
    } catch {
      setError("Clipboard permission denied. Paste manually with Ctrl+V.");
    }
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
        setActiveTab("benchmarks");
      }
    } finally {
      setAuditLoading(false);
    }
  };

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
      />

      <main className="mx-auto max-w-6xl space-y-4 px-4 py-5 sm:px-6">
        {activeTab === "benchmarks" ? (
          <BenchmarkView benchmarks={benchmarks} auditProof={auditProof} />
        ) : (
          <>
            <ScannerInput
              text={text}
              onChangeText={setText}
              sender={sender}
              onChangeSender={setSender}
              onScan={(t) => runScan(t, "text")}
              onMutate={handleMutate}
              onPasteAndScan={handlePasteAndScan}
              loading={loading}
              error={error}
              runtimeMode={runtimeMode}
            />
            <VerdictHero result={result} />
            <SmsBubbleInspector rawText={text} result={result} />
            <ForensicDrawers result={result} mutationData={mutationData} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
