import asyncio
import base64
import json
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import aiohttp

CHROMIUM_NAMES = ["chromium-browser", "chromium", "google-chrome", "google-chrome-stable"]
REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
SWIFTSHADER_FLAGS = ["--disable-gpu", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"]
GPU_FLAGS = ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist"]   # Vulkan loses the context on the first frame
STARTUP_TIMEOUT_S = 20.0
CLOSE_TIMEOUT_S = 10
POLL_S = 0.1


def find_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def describe_exception(details):
    text = details.get("exception", {}).get("description") or details["text"]
    if "url" in details:
        text += f" ({details['url']}:{details['lineNumber'] + 1})"
    return text


class Browser:
    def __init__(self, width=1400, height=900, gpu=False):
        self.width, self.height = width, height
        self.gpu = gpu
        self.console = []
        self.process = None
        self.profile = None
        self.http = None
        self.ws = None
        self.pump_task = None
        self.pending = {}
        self.next_id = 0

    async def __aenter__(self):
        chromium = next(filter(None, map(shutil.which, CHROMIUM_NAMES)), None)
        if chromium is None:
            raise RuntimeError(f"no browser on PATH, looked for {CHROMIUM_NAMES}")
        port = find_free_port()
        OUTPUTS.mkdir(exist_ok=True)
        self.profile = tempfile.mkdtemp(prefix="chrome-profile-", dir=OUTPUTS)
        self.process = subprocess.Popen([
            chromium, "--headless=new", *(GPU_FLAGS if self.gpu else SWIFTSHADER_FLAGS),
            "--no-sandbox", f"--remote-debugging-port={port}",
            f"--window-size={self.width},{self.height}", f"--user-data-dir={self.profile}", "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.http = aiohttp.ClientSession()
        ws_url = None
        deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT_S
        while ws_url is None:
            if asyncio.get_running_loop().time() > deadline:
                await self.__aexit__(None, None, None)
                raise RuntimeError(f"chromium debug endpoint on port {port} did not come up in {STARTUP_TIMEOUT_S}s")
            try:
                async with self.http.get(f"http://127.0.0.1:{port}/json") as r:
                    pages = [t for t in await r.json() if t["type"] == "page"]
            except aiohttp.ClientConnectorError:
                pages = []
            if pages:
                ws_url = pages[0]["webSocketDebuggerUrl"]
            else:
                await asyncio.sleep(POLL_S)
        self.ws = await self.http.ws_connect(ws_url, max_msg_size=256 * 1024 * 1024)
        self.pump_task = asyncio.create_task(self.pump())
        await self.send("Runtime.enable")
        await self.send("Page.enable")
        await self.send("Emulation.setDeviceMetricsOverride", {
            "width": self.width, "height": self.height, "deviceScaleFactor": 1, "mobile": False,
        })
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.ws is not None:
            closing = asyncio.ensure_future(self.send("Browser.close"))
            if not (await asyncio.wait({closing}, timeout=CLOSE_TIMEOUT_S))[0]:
                closing.cancel()
                self.process.kill()
        elif self.process.poll() is None:
            self.process.terminate()
        if self.pump_task is not None:
            self.pump_task.cancel()
        if self.ws is not None:
            await self.ws.close()
        if self.http is not None:
            await self.http.close()
        self.process.wait(timeout=10)
        shutil.rmtree(self.profile)

    async def pump(self):
        async for msg in self.ws:
            data = json.loads(msg.data)
            if "id" in data:
                self.pending.pop(data["id"]).set_result(data)
            elif data["method"] == "Runtime.consoleAPICalled":
                params = data["params"]
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in params["args"])
                self.console.append({"level": params["type"], "text": text})
            elif data["method"] == "Runtime.exceptionThrown":
                self.console.append({"level": "error", "text": describe_exception(data["params"]["exceptionDetails"])})

    async def send(self, method, params=None):
        self.next_id += 1
        future = asyncio.get_running_loop().create_future()
        self.pending[self.next_id] = future
        await self.ws.send_str(json.dumps({"id": self.next_id, "method": method, "params": params or {}}))
        reply = await future
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error']}")
        return reply["result"]

    async def navigate(self, url):
        result = await self.send("Page.navigate", {"url": url})
        if "errorText" in result:
            raise RuntimeError(f"navigate {url}: {result['errorText']}")
        await self.wait_for("document.readyState === 'complete'", timeout=STARTUP_TIMEOUT_S)

    async def evaluate(self, js):
        result = await self.send("Runtime.evaluate", {
            "expression": js, "returnByValue": True, "awaitPromise": True,
        })
        if "exceptionDetails" in result:
            raise RuntimeError(f"evaluate failed: {describe_exception(result['exceptionDetails'])}\n  in: {js}")
        return result["result"].get("value")

    async def wait_for(self, js, timeout):
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            value = await self.evaluate(js)
            if value:
                return value
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(f"wait_for timed out after {timeout}s, last value {value!r}: {js}")
            await asyncio.sleep(POLL_S)

    async def screenshot(self, path):
        shot = await self.send("Page.captureScreenshot", {"format": "png"})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(base64.b64decode(shot["data"]))
