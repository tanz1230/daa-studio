"""A minimal Chrome DevTools Protocol driver for the UI smoke test: headless Chrome + websocket-client.

No Selenium or Playwright needed. WebGL runs on SwiftShader (CPU), so it works on a busy GPU box.
"""
from __future__ import annotations

import base64
import itertools
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request

import websocket

_KEYS = {  # key -> (code, windowsVirtualKeyCode, text)
    "Enter": ("Enter", 13, "\r"), "Escape": ("Escape", 27, ""), "Tab": ("Tab", 9, ""),
    "Delete": ("Delete", 46, ""), "Backspace": ("Backspace", 8, ""), "Home": ("Home", 36, ""),
    "End": ("End", 35, ""), "ArrowLeft": ("ArrowLeft", 37, ""), "ArrowUp": ("ArrowUp", 38, ""),
    "ArrowRight": ("ArrowRight", 39, ""), "ArrowDown": ("ArrowDown", 40, ""), " ": ("Space", 32, " "),
}


class Chrome:
    def __init__(self, width=1600, height=1000, port=9333, chrome="google-chrome", scale=1):
        self.profile = tempfile.mkdtemp(prefix="daa-ui-")
        self.proc = subprocess.Popen(
            [chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={self.profile}",
             f"--window-size={width},{height}", "--no-first-run", "--no-default-browser-check",
             "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            # Chrome must not see a conda LD_LIBRARY_PATH (needed for scipy here): it would load
            # conda's libffi and die at startup with "undefined symbol: ffi_type_uint32"
            env={k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"})
        page = None
        for _ in range(600):                         # up to a minute on a loaded machine
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        if page is None:
            self.proc.kill()
            shutil.rmtree(self.profile, ignore_errors=True)
            raise RuntimeError("Chrome did not start")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120, suppress_origin=True)
        self._ids = itertools.count(1)
        self.console: list = []
        for m in ("Page.enable", "Runtime.enable", "Log.enable"):
            self.send(m)
        self.send("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale,
                  mobile=False)

    # ------------------------------------------------------------------ protocol
    def send(self, method, **params):
        i = next(self._ids)
        self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == i:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self._event(msg)

    def _event(self, msg):
        m, p = msg.get("method"), msg.get("params", {})
        if m == "Runtime.consoleAPICalled":
            self.console.append((p["type"], " ".join(str(a.get("value", a.get("description", ""))) for a in p["args"])))
        elif m == "Runtime.exceptionThrown":
            d = p["exceptionDetails"]
            self.console.append(("exception", (d.get("exception") or {}).get("description") or d.get("text")))
        elif m == "Log.entryAdded":
            e = p["entry"]
            self.console.append((e["level"], f"{e.get('text', '')} {e.get('url', '')}".strip()))

    def sleep(self, s):
        end = time.time() + s
        self.ws.settimeout(0.05)
        try:
            while time.time() < end:
                try:
                    self._event(json.loads(self.ws.recv()))
                except websocket.WebSocketTimeoutException:
                    pass
        finally:
            self.ws.settimeout(120)

    # ------------------------------------------------------------------ page
    def goto(self, url, wait=1.5):
        self.send("Page.navigate", url=url)
        self.sleep(wait)

    def js(self, expr):
        r = self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError((d.get("exception") or {}).get("description") or d.get("text"))
        return r["result"].get("value")

    def wait_for(self, expr, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.js(expr):
                return True
            self.sleep(0.1)
        raise TimeoutError(expr)

    def shot(self, path):
        with open(path, "wb") as f:
            f.write(base64.b64decode(self.send("Page.captureScreenshot", format="png")["data"]))

    # ------------------------------------------------------------------ input
    def key(self, key, shift=False, ctrl=False):
        mods = (8 if shift else 0) | (2 if ctrl else 0)
        if key in _KEYS:
            code, vk, text = _KEYS[key]
        else:
            code = f"Key{key.upper()}" if key.isalpha() else ""
            vk, text = ord(key.upper()), ("" if ctrl else key)
        base = dict(key=key, code=code, windowsVirtualKeyCode=vk, nativeVirtualKeyCode=vk, modifiers=mods)
        self.send("Input.dispatchKeyEvent", type="keyDown" if text else "rawKeyDown", text=text, **base)
        self.send("Input.dispatchKeyEvent", type="keyUp", **base)
        self.sleep(0.05)

    def type(self, text):
        self.send("Input.insertText", text=text)

    def mouse(self, kind, x, y, button="left", shift=False, buttons=None):
        self.send("Input.dispatchMouseEvent", type=kind, x=x, y=y, button=button, clickCount=1,
                  modifiers=8 if shift else 0, buttons=1 if buttons is None and button == "left" else (buttons or 0))

    def click(self, x, y, shift=False):
        self.mouse("mouseMoved", x, y, button="none", buttons=0)
        self.mouse("mousePressed", x, y, shift=shift)
        self.mouse("mouseReleased", x, y, shift=shift, buttons=0)
        self.sleep(0.1)

    def drag(self, x0, y0, x1, y1, steps=12, shift=False):
        self.mouse("mouseMoved", x0, y0, button="none", buttons=0)
        self.mouse("mousePressed", x0, y0, shift=shift)
        for i in range(1, steps + 1):
            self.mouse("mouseMoved", x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps, shift=shift)
            self.sleep(0.02)
        self.mouse("mouseReleased", x1, y1, shift=shift, buttons=0)
        self.sleep(0.1)

    def center(self, selector):
        r = self.js(f"(() => {{ const n = document.querySelector({json.dumps(selector)}); if (!n) return null;"
                    f" const r = n.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }})()")
        if r is None:
            raise LookupError(selector)
        return r

    def click_sel(self, selector):
        x, y = self.center(selector)
        self.click(x, y)

    def click_text(self, text, selector="button"):
        """Click the first element matching `selector` whose text contains `text`."""
        r = self.js(f"(() => {{ const n = [...document.querySelectorAll({json.dumps(selector)})].find((n) =>"
                    f" n.textContent.includes({json.dumps(text)}) && n.offsetParent !== null); if (!n) return null;"
                    f" const r = n.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }})()")
        if r is None:
            raise LookupError(f"{selector}:{text}")
        self.click(*r)

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            shutil.rmtree(self.profile, ignore_errors=True)
