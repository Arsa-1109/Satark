"""Thread-safe Windows ctypes Clipboard Guardian poller (SDD §2.5)."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import sys
import threading
import time
from typing import Callable, Awaitable

CF_UNICODETEXT = 13


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
                                self._dispatch(text)
                except Exception:
                    pass
            time.sleep(self.poll_interval)
