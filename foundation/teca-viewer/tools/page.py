import asyncio
import base64
import contextlib
import json
import subprocess
import sys
import time

import aiohttp

from browser import find_free_port

VOLUME = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
SURFACE = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20231221180251/mesh/"
           "20231221180251-on-20260411134726-2.4um.tifxyz/")
VOLUME_PATH = "/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"   # S3 mirrors change the host, not the path
CENTER_XYZ = "20510.63, 13999.20, 43903.17"   # surface grid cell (row 812, col 1040) of its x/y/z.tif: all three planes cut the surface
WIDTH, HEIGHT = 1440, 1000
NEUTRAL_XY = (150, 950)
QUIET_S = 0.5             # a key must hold this long; above config.js's 320 ms SETTLE_DELAY_MS
POLL_S = 0.1
TIMEOUT_S = 180.0
SERVE_TIMEOUT_S = 10.0
BOOT_TIMEOUT_S = 20.0     # main.js installs __tecaDebug 1.1 s after the load, at most 1.4 s in 60 boots 3 at once
BLANK_COLORS = 16
KEYS = {"Enter": ("Enter", 13), "f": ("KeyF", 70), "u": ("KeyU", 85), "ArrowRight": ("ArrowRight", 39), "Tab": ("Tab", 9),
        "a": ("KeyA", 65), "c": ("KeyC", 67), "B": ("KeyB", 66), "]": ("BracketRight", 221), "2": ("Digit2", 50),
        "z": ("KeyZ", 90), "K": ("KeyK", 75), "L": ("KeyL", 76), "F": ("KeyF", 70), "O": ("KeyO", 79), "D": ("KeyD", 68),
        "Escape": ("Escape", 27), ",": ("Comma", 188), "e": ("KeyE", 69), "'": ("Quote", 222),
        ".": ("Period", 190), "w": ("KeyW", 87), "ArrowDown": ("ArrowDown", 40), "ArrowUp": ("ArrowUp", 38),
        "Delete": ("Delete", 46), "Backspace": ("Backspace", 8), "v": ("KeyV", 86), ";": ("Semicolon", 186),
        "s": ("KeyS", 83), "n": ("KeyN", 78), "m": ("KeyM", 77), "t": ("KeyT", 84), "x": ("KeyX", 88),
        "6": ("Digit6", 54), "1": ("Digit1", 49), "7": ("Digit7", 55),
        "4": ("Digit4", 52), " ": ("Space", 32), "End": ("End", 35), "ArrowLeft": ("ArrowLeft", 37)}
SHIFT, CTRL = 8, 2

STATE = """(() => {
  const d = window.__tecaDebug;
  if (d === undefined) return null;
  const slices = (g) => ({
    pending: g.pendingFetches, upload: g.uploadQueue,
    panes: Object.entries(g.panes).filter(([, p]) => p.visible).map(([name, p]) =>
      [name, p.scaleIndex, p.token === p.plannedToken, p.levels.map(l => [l.scaleIndex, l.available, l.planned])]),
  });
  const g = d.slicedPlaneStats();
  const s = d.surfaceStats();
  return {
    ct: g === null ? null : slices(g),
    layers: d.layerStats().map(slices),
    lines: Object.fromEntries(Object.entries(d.overlay2dStats()).map(([pane, o]) => [pane, o.lineVertices])),
    cutting: Object.values(d.overlay2dStats()).reduce((total, o) => total + o.cutting, 0),
    surface: s === null ? null : [s.pendingChunks, s.queuedChunkUploads, s.covered, s.drawn, s.planned],
    frames: d.renderStats().orthogonalFrames,
  };
})()"""
PIXELS = """(async (png) => {
  const blob = await (await fetch("data:image/png;base64," + png)).blob();
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", rgba));
  const colors = new Set();
  for (let y = bitmap.height >> 2; y < (3 * bitmap.height) >> 2; y++)
    for (let x = bitmap.width >> 2; x < (3 * bitmap.width) >> 2; x++) {
      const i = 4 * (y * bitmap.width + x);
      colors.add(rgba[i] | (rgba[i + 1] << 8) | (rgba[i + 2] << 16));
    }
  return { sha256: [...digest].map(b => b.toString(16).padStart(2, "0")).join(""), centralColors: colors.size };
})"""
HIDE_CROSSHAIR = """new Promise((resolve) => {
  const shell = document.querySelector('.viewer-shell');
  const shown = !shell.classList.contains('crosshair-hidden');
  shell.classList.add('crosshair-hidden');
  requestAnimationFrame(() => requestAnimationFrame(() => resolve(shown)));
})"""
SHOW_CROSSHAIR = "document.querySelector('.viewer-shell').classList.remove('crosshair-hidden'), true"
PANES = """(() => Object.fromEntries(["xy", "zy", "xz", "3d", "surface"].map(name => {
  const r = document.querySelector("#viewer-" + name).getBoundingClientRect();
  return [name, { x: r.x, y: r.y, width: r.width, height: r.height }];
}).filter(([, r]) => r.width > 0 && r.height > 0)))()"""
ERRORS = "[...document.querySelectorAll('[role=status].error:not([hidden])')].map(e => `#${e.id}: ${e.textContent}`)"
BOOTED = ("window.__tecaDebug !== undefined"
          " || document.querySelector('#startup-title').textContent !== 'Starting viewer…'")
STARTUP = "document.querySelector('#startup-message').innerText.replaceAll('\\n', ' ')"
TAB = """((selector) => {   // the selector of the tab button of the side panel's tab that holds `selector`
  const element = document.querySelector(selector);
  const left = element.closest("[data-lpanel]");
  return left !== null ? `[data-ltab="${left.dataset.lpanel}"]` : `[data-rtab="${element.closest("[data-rpanel]").dataset.rpanel}"]`;
})"""


def slices_key(slices):
    if slices is None or slices["pending"] != 0 or slices["upload"] != 0 or not slices["panes"]:
        return None
    for _, _, current, levels in slices["panes"]:
        if not current or not any(planned > 0 for _, _, planned in levels) or any(available < planned for _, available, planned in levels):
            return None
    return json.dumps(slices["panes"])


def ct_key(s):
    return None if s is None else slices_key(s["ct"])


def lines_key(s):
    return None if s is None or s["cutting"] != 0 or sum(s["lines"].values()) == 0 else json.dumps(s["lines"])


def surface_key(s):
    return None if s is None or s["surface"] != [0, 0, True, True, True] else "complete"


def session_link(session):
    text = json.dumps(session, separators=(",", ":"), ensure_ascii=False)
    return "#session=" + base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


@contextlib.asynccontextmanager
async def serve(tree):
    port = find_free_port()
    root = f"http://127.0.0.1:{port}/"
    server = subprocess.Popen([sys.executable, str(tree / "serve.py"), "--port", str(port)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as http:
            deadline = time.monotonic() + SERVE_TIMEOUT_S
            while True:
                try:
                    async with http.get(root) as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientConnectorError:
                    pass
                if time.monotonic() > deadline or server.poll() is not None:
                    raise RuntimeError(f"{tree / 'serve.py'} did not serve {root} in {SERVE_TIMEOUT_S} s (exit {server.poll()})")
                await asyncio.sleep(POLL_S)
        yield root
    finally:
        server.kill()
        server.wait()


class Page:
    def __init__(self, browser, root):
        self.browser = browser
        self.root = root
        self.navigated_at = None

    async def open(self, query):
        # The load event comes before main.js's modules, which startup.js imports, and a navigation among them cancels
        # requests that the browser never reports as ended. A link with a fragment is opened from about:blank, because a
        # navigation that differs from the page's address in its fragment alone stays in the same document and loads
        # nothing.
        self.navigated_at = time.monotonic()
        if "#" in query:
            await self.browser.navigate("about:blank")
        await self.browser.navigate(self.root + query)
        try:
            await self.browser.wait_for(BOOTED, timeout=BOOT_TIMEOUT_S)
        except TimeoutError as error:
            open_urls = [r["url"].removeprefix(self.root) for r in self.browser.requests.values()
                         if r["url"].startswith(self.root) and r["end"] is None]
            raise TimeoutError(f"{error}; the page shows {await self.evaluate(STARTUP)!r}, requests open {open_urls}") from error
        if not await self.evaluate("window.__tecaDebug !== undefined"):
            raise RuntimeError(f"the page did not start: {await self.evaluate(STARTUP)!r}")
        await self.mouse("mouseMoved", *NEUTRAL_XY)

    async def open_session(self, session):
        await self.open(session_link(session))

    async def address(self):
        return (await self.evaluate("location.href")).removeprefix(self.root)

    async def address_session(self):
        token = await self.evaluate("new URLSearchParams(location.hash.slice(1)).get('session')")
        if token is None:
            raise RuntimeError(f"the address holds no #session=: {(await self.address())[:80]}…")
        return json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))

    async def evaluate(self, js):
        return await self.browser.evaluate(js)

    async def config(self, name):
        found = await self.evaluate(f"import('./src/config.js').then((config) => "
                                    f"Object.hasOwn(config, {json.dumps(name)}) ? [config[{json.dumps(name)}]] : null)")
        if found is None:
            raise KeyError(f"src/config.js exports no {name}")
        return found[0]

    async def state(self):
        return await self.evaluate(STATE)

    async def settle(self, t0, keys, firsts=None):
        firsts = firsts or {}
        stretch = {name: (None, None) for name in keys}
        first_at = {name: None for name in firsts}
        while True:
            s = await self.state()
            now = time.monotonic()
            in_flight = [r["url"] for r in self.browser.requests.values() if r["start"] >= self.navigated_at and r["end"] is None
                         and not (r["url"].startswith(self.root) and r["url"].endswith(".js"))]
            for name, first in firsts.items():
                if first_at[name] is None and first(s):
                    first_at[name] = round(now - t0, 2)
            for name, key in keys.items():
                value = key(s) if not in_flight else None
                if value is None:
                    stretch[name] = (None, None)
                elif value != stretch[name][0]:
                    stretch[name] = (value, now)
            if all(since is not None and now - since >= QUIET_S for _, since in stretch.values()):
                missing = [name for name, at in first_at.items() if at is None]
                if missing:
                    raise RuntimeError(f"settled without {missing}; last state {json.dumps(s)}")
                return {name: round(since - t0, 2) for name, (_, since) in stretch.items()}, first_at
            if now - t0 > TIMEOUT_S:
                raise TimeoutError(f"not settled after {TIMEOUT_S} s: {[n for n, (_, since) in stretch.items() if since is None]} "
                                   f"unmet, {len(in_flight)} requests in flight {in_flight[:3]}, last state {json.dumps(s)}, "
                                   f"errors {await self.errors()}")
            await asyncio.sleep(POLL_S)

    async def wait_text(self, text, done, timeout):
        deadline = time.monotonic() + timeout
        while not done(value := await self.evaluate(text)):
            if time.monotonic() > deadline:
                raise TimeoutError(f"after {timeout} s, {text} is {value!r}")
            await asyncio.sleep(POLL_S)
        return value

    async def key(self, name, repeat=False, down_only=False, shift=False, ctrl=False):
        code, vk = KEYS[name]
        text = {"text": name} if len(name) == 1 else {}
        modifiers = (SHIFT if shift else 0) | (CTRL if ctrl else 0)
        await self.browser.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": name, "code": code, "modifiers": modifiers,
                                                           "windowsVirtualKeyCode": vk, "autoRepeat": repeat, **text})
        if not down_only:
            await self.browser.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": name, "code": code, "modifiers": modifiers,
                                                               "windowsVirtualKeyCode": vk})

    async def mouse(self, kind, x, y, **extra):
        await self.browser.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, **extra})

    async def click(self, selector):
        # The middle is read after the next frame, so a panel that lays out again in that frame has already moved the
        # element.
        x, y = await self.evaluate(f"new Promise((resolve) => requestAnimationFrame(() => {{ const e = document.querySelector({json.dumps(selector)}); "
                                   "e.scrollIntoView({ block: 'center' }); const r = e.getBoundingClientRect(); resolve([r.x + r.width / 2, r.y + r.height / 2]) }))")
        await self.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
        await self.mouse("mouseReleased", x, y, button="left", buttons=0, clickCount=1)

    async def place(self, selector):
        return await self.evaluate(f"(() => {{ const tab = document.querySelector({TAB}({json.dumps(selector)})); "
                                   "return tab.dataset.ltab ?? tab.title })()")

    async def reveal(self, selector):
        await self.click(await self.evaluate(f"{TAB}({json.dumps(selector)})"))

    async def type_into(self, element_id, text):
        await self.evaluate(f"(() => {{ const e = document.querySelector('#{element_id}'); e.focus(); e.select(); return true }})()")
        await self.browser.send("Input.insertText", {"text": text})

    async def choose(self, selector, value=None, checked=None):
        await self.evaluate(f"""(() => {{ const e = document.querySelector({json.dumps(selector)});
          {'' if value is None else f'e.value = {json.dumps(value)};'}
          {'' if checked is None else f'e.checked = {json.dumps(checked)};'}
          e.dispatchEvent(new Event('input', {{ bubbles: true }})); e.dispatchEvent(new Event('change', {{ bubbles: true }})); return true }})()""")

    async def choose_file(self, selector, path):
        element = await self.browser.send("Runtime.evaluate", {"expression": f"document.querySelector({json.dumps(selector)})"})
        await self.browser.send("DOM.setFileInputFiles", {"files": [str(path)], "objectId": element["result"]["objectId"]})

    async def download(self, selector, name):
        path = self.browser.downloads / name
        t0 = time.monotonic()
        await self.click(selector)
        while not path.exists():
            if time.monotonic() - t0 > TIMEOUT_S:
                raise TimeoutError(f"{selector} downloaded no {name} in {TIMEOUT_S} s: {sorted(self.browser.downloads.glob('*'))}")
            await asyncio.sleep(POLL_S)
        return path

    async def add_surface(self):
        await self.reveal("#overlay-add")
        await self.type_into("overlay-path", SURFACE)
        await self.click("#overlay-add")
        await self.mouse("mouseMoved", *NEUTRAL_XY)

    async def panes(self):
        return await self.evaluate(PANES)

    async def pane_center(self, pane):
        rect = (await self.panes())[pane]
        return rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2

    async def picture(self, rect):
        # Chrome's compositor blends the crosshair over the panes, so two pictures of the same panes can differ by a
        # colour level along it.
        shown = await self.evaluate(HIDE_CROSSHAIR)
        png = await self.browser.screenshot(rect)
        if shown:
            await self.evaluate(SHOW_CROSSHAIR)
        return png, await self.evaluate(f"{PIXELS}({json.dumps(base64.b64encode(png).decode())})")

    async def extension_viewer(self):
        found = await self.browser.send("Runtime.queryObjects", {"prototypeObjectId": (await self.browser.send(
            "Runtime.evaluate", {"expression": "Object.prototype"}))["result"]["objectId"]})
        result = await self.browser.send("Runtime.callFunctionOn", {
            "objectId": found["objects"]["objectId"], "returnByValue": True,
            "functionDeclaration": "function () { window.__viewer = this.find((object) => ['openVolume', "
                                   "'addLayer', 'focusPoint', 'setStatus'].every((name) => "
                                   "typeof object[name] === 'function')); return window.__viewer !== undefined }"})
        if not result["result"]["value"]:
            raise RuntimeError("the page's heap holds no object with the methods of the extensions' viewer")

    def expected(self, fragment):
        made = [entry for entry in self.browser.console if entry["level"] == "error" and fragment in entry["text"]]
        self.browser.console[:] = [entry for entry in self.browser.console if entry not in made]
        return len(made)

    async def errors(self):
        return [c["text"] for c in self.browser.console if c["level"] == "error"] + await self.evaluate(ERRORS)
