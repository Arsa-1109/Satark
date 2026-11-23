import { CheckCircle2, XCircle } from "lucide-react";

export function BenchmarkView({ benchmarks, auditProof }) {
  if (!benchmarks) {
    return (
      <div className="rounded-lg border border-[#E4E4E0] bg-white p-6 text-sm text-[#52525B]">
        Loading held-out evaluation benchmarks...
      </div>
    );
  }

  const full = benchmarks.full_onnx_hybrid || {};
  const lite = benchmarks.lite_fallback || {};
  const gates = benchmarks.gates || {};

  const cards = [
    { label: "Unseen Binary F1 (G1)", full: full.unseen_f1, lite: lite.unseen_f1, target: ">= 0.94" },
    { label: "Hard-Negative FPR (G2)", full: `${full.hard_neg_fpr_pct}%`, lite: `${lite.hard_neg_fpr_pct}%`, target: "<= 2.0%" },
    { label: "5-Class Macro-F1 (G3)", full: full.macro_f1_5class, lite: lite.macro_f1_5class, target: ">= 0.88" },
    { label: "Adversarial Recall (G4)", full: full.adversarial_recall, lite: lite.adversarial_recall, target: ">= 0.90" },
    { label: "Calibration ECE / Brier (G5)", full: `${full.ece} / ${full.brier_score}`, lite: `${lite.ece} / ${lite.brier_score}`, target: "<= 0.06 / <= 0.08" },
    { label: "Median CPU Latency (G6)", full: `${full.median_latency_ms} ms`, lite: `${lite.median_latency_ms} ms`, target: "< 80 ms / < 10 ms" },
  ];

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-[#E4E4E0] bg-white p-4 sm:p-5">
        <h2 className="text-balance text-sm font-bold uppercase tracking-wider text-[#111111]">
          Held-Out Verification Gates (G1–G6)
        </h2>
        <p className="text-pretty mt-1 text-xs text-[#52525B]">
          Measured on strictly deduplicated unseen, TRAI DLT hard-negative, and adversarial splits.
        </p>

        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {cards.map((c) => (
            <div key={c.label} className="rounded border border-[#E4E4E0] bg-[#F7F7F5] p-3">
              <div className="text-xs font-semibold text-[#52525B]">{c.label}</div>
              <div className="font-mono-tabular mt-1 text-lg font-extrabold text-[#111111]">
                {c.full}
              </div>
              <div className="font-mono-tabular mt-1 flex items-center justify-between text-[11px] text-[#52525B]">
                <span>Lite/PWA: {c.lite}</span>
                <span>Target: {c.target}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 flex flex-wrap gap-2 border-t border-[#E4E4E0] pt-3">
          {Object.entries(gates).map(([k, passed]) => (
            <span
              key={k}
              className="font-mono-tabular inline-flex items-center gap-1 rounded border border-[#E4E4E0] bg-[#F7F7F5] px-2 py-1 text-[11px] font-semibold"
            >
              {passed ? (
                <CheckCircle2 className="size-3.5 text-[#047857]" />
              ) : (
                <XCircle className="size-3.5 text-[#C81E1E]" />
              )}
              {k}
            </span>
          ))}
        </div>
      </section>

      {auditProof && (
        <section className="rounded-lg border border-[#047857]/40 bg-white p-4 sm:p-5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[#047857]">
            Subprocess Zero-Network Audit Proof (Gate G8)
          </h3>
          <pre className="font-mono-tabular mt-2 overflow-x-auto rounded border border-[#E4E4E0] bg-[#F7F7F5] p-3 text-xs text-[#111111]">
            {JSON.stringify(auditProof, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}
