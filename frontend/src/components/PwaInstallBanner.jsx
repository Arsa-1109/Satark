import { useEffect, useState } from "react";
import { Download, Smartphone, X } from "lucide-react";

const DISMISS_KEY = "satark_install_dismissed_v1";

export function PwaInstallBanner() {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [isIosHint, setIsIosHint] = useState(false);
  const [dismissed, setDismissed] = useState(() => {
    try {
      return window.localStorage.getItem(DISMISS_KEY) === "1";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    const isStandalone =
      window.matchMedia?.("(display-mode: standalone)").matches ||
      window.navigator.standalone === true;
    if (isStandalone) return;

    const ua = window.navigator.userAgent || "";
    const isIos = /iphone|ipad|ipod/i.test(ua);
    if (isIos) {
      setIsIosHint(true);
    }

    const onBeforeInstall = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
    };

    window.addEventListener("beforeinstallprompt", onBeforeInstall);
    return () => window.removeEventListener("beforeinstallprompt", onBeforeInstall);
  }, []);

  if (dismissed || (!deferredPrompt && !isIosHint)) return null;

  const handleDismiss = () => {
    setDismissed(true);
    try {
      window.localStorage.setItem(DISMISS_KEY, "1");
    } catch {
      // ignore
    }
  };

  const handleInstall = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    try {
      await deferredPrompt.userChoice;
    } finally {
      setDeferredPrompt(null);
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#047857]/30 bg-[#047857]/[0.06] px-4 py-3 text-xs text-[#111111]">
      <div className="flex items-center gap-2.5">
        <Smartphone className="size-4 shrink-0 text-[#047857]" />
        {deferredPrompt ? (
          <span>
            <strong className="font-semibold">Install Offline App:</strong> Enables{" "}
            <span className="font-semibold">Share from WhatsApp / SMS → Satark</span> even in Airplane Mode.
          </span>
        ) : (
          <span>
            <strong className="font-semibold">Add to Home Screen:</strong> Tap Share →{" "}
            <span className="font-semibold">Add to Home Screen</span> to use Satark offline anytime.
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        {deferredPrompt && (
          <button
            type="button"
            onClick={handleInstall}
            className="inline-flex min-h-[38px] items-center gap-1.5 rounded-md bg-[#111111] px-3 py-1.5 text-xs font-semibold text-white hover:bg-[#27272A]"
          >
            <Download className="size-3.5" />
            Install App
          </button>
        )}
        <button
          type="button"
          onClick={handleDismiss}
          aria-label="Dismiss install tip"
          className="flex size-9 items-center justify-center rounded text-[#52525B] hover:bg-white hover:text-[#111111]"
        >
          <X className="size-4" />
        </button>
      </div>
    </div>
  );
}
