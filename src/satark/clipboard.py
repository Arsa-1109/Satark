"""Thread-safe Windows ctypes Clipboard Guardian poller (SDD §2.5)."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import re
import sys
import threading
import time

CF_UNICODETEXT = 13

_HIGH_SIGNAL_RE = re.compile(r"(?:upi://|https?://|bit\.ly|tinyurl|\.apk\b)", re.IGNORECASE)
_LOCAL_PATH_RE = re.compile(r"^(?:[A-Za-z]:\\|/(?:usr|var|etc|home|Users)/|\\\\)")
_CODE_LINE_RE = re.compile(
    r"^(?:import\s+|from\s+\S+\s+import\s+|def\s+|class\s+|function\s+|const\s+|let\s+|var\s+|SELECT\s+|#include\s+)"
)


def is_scannable_clipboard_text(text: str) -> bool:
    """Filter out trivial or non-message clipboard copies (short snippets, paths, code)."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 10_000:
        return False

    if _HIGH_SIGNAL_RE.search(stripped) and len(stripped) >= 10:
        return True

    if len(stripped) < 15:
        return False

    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) == 1 and _LOCAL_PATH_RE.match(lines[0]):
        return False

    if stripped.startswith("{") and stripped.endswith("}"):
        return False

    code_lines = sum(1 for ln in lines if _CODE_LINE_RE.match(ln))
    if code_lines >= 2:
        return False

    return True


def _read_windows_clipboard() -> str | None:
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_bool
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = ctypes.c_bool
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_bool

    if not user32.OpenClipboard(None):
        return None
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return None
        try:
            text = ctypes.wstring_at(ptr)
            return text
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


class ClipboardGuardian:
    """Polls Windows clipboard every 400ms on a daemon thread and notifies subscribers."""

    def __init__(self, poll_interval: float = 0.4) -> None:
        self.poll_interval = poll_interval
        self.supported = sys.platform == "win32"
        self.enabled = False
        self.only_alert_scam = False
        self.smart_filter = True
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_hash: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=32)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if not self.supported:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        # Seed initial clipboard hash so we don't fire on pre-existing clipboard content
        initial = _read_windows_clipboard()
        if initial:
            self._last_hash = hashlib.sha256(initial.encode("utf-8", errors="ignore")).hexdigest()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="satark-clipboard")
        self._thread.start()

    def set_enabled(self, enabled: bool) -> bool:
        if not self.supported:
            self.enabled = False
            return False
        if enabled and not self.enabled:
            # Refresh hash on enable so existing clipboard isn't re-scanned immediately
            current = _read_windows_clipboard()
            if current:
                self._last_hash = hashlib.sha256(current.encode("utf-8", errors="ignore")).hexdigest()
        self.enabled = enabled
        return self.enabled

    def set_config(
        self,
        enabled: bool | None = None,
        only_alert_scam: bool | None = None,
        smart_filter: bool | None = None,
    ) -> dict[str, bool]:
        if enabled is not None:
            self.set_enabled(enabled)
        if only_alert_scam is not None:
            self.only_alert_scam = bool(only_alert_scam)
        if smart_filter is not None:
            self.smart_filter = bool(smart_filter)
        return {
            "enabled": self.enabled,
            "supported": self.supported,
            "only_alert_scam": self.only_alert_scam,
            "smart_filter": self.smart_filter,
        }

    def stop(self) -> None:
        self._stop_event.set()
        self.enabled = False

    def _dispatch(self, text: str) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            def _put(queue: asyncio.Queue[str] = q, payload: str = text) -> None:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(payload)
            self._loop.call_soon_threadsafe(_put)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self.enabled:
                try:
                    text = _read_windows_clipboard()
                    if text and text.strip():
                        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                        if digest != self._last_hash:
                            self._last_hash = digest
                            if 1 <= len(text) <= 10_000:
                                if not self.smart_filter or is_scannable_clipboard_text(text):
                                    self._dispatch(text)
                except Exception:
                    pass
            time.sleep(self.poll_interval)
