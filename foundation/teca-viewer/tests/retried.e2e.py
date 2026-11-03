import asyncio
import json
import re
import time

from aiohttp import web

from browser import find_free_port
from page import POLL_S, TIMEOUT_S, ct_key

CHUNKS = 4
CHUNK = 64
CHUNK_KEY = re.compile(r"^/retried\.zarr/0/(\d)/(\d)/(\d)$")
CORS = {"Access-Control-Allow-Origin": "*"}   # on the 503 too, as S3 sends it: without it the page sees a network error
LEVEL = {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}
ATTRS = {"multiscales": [{"axes": [{"name": axis, "type": "space"} for axis in "zyx"], "datasets": [LEVEL]}]}
ARRAY = {"zarr_format": 2, "shape": [CHUNKS * CHUNK] * 3, "chunks": [CHUNK] * 3, "dtype": "|u1", "compressor": None,
         "filters": None, "order": "C", "fill_value": 0, "dimension_separator": "/"}
SLOW_DOWN = "<Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message></Error>"


async def check(page, out):
    asked = {}

    async def answer(request):
        asked[request.path] = asked.get(request.path, 0) + 1
        if asked[request.path] == 1:
            return web.Response(status=503, text=SLOW_DOWN, content_type="application/xml", headers=CORS)
        if request.path == "/retried.zarr/.zattrs":
            return web.json_response(ATTRS, headers=CORS)
        if request.path == "/retried.zarr/0/.zarray":
            return web.json_response(ARRAY, headers=CORS)
        if chunk := CHUNK_KEY.match(request.path):
            value = 40 + 25 * (sum(map(int, chunk.groups())) % 8)
            return web.Response(body=bytes([value]) * CHUNK ** 3, content_type="application/octet-stream", headers=CORS)
        return web.Response(status=404, headers=CORS)

    app = web.Application()
    app.router.add_get("/{path:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    port = find_free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        t0 = time.monotonic()
        await page.open(f"?volume=http://127.0.0.1:{port}/retried.zarr/")
        while ct_key(await page.state()) is None:
            if await page.errors():
                return [f"the page showed an error before its panes were complete; asked {json.dumps(asked)}"]
            if time.monotonic() - t0 > TIMEOUT_S:
                return [f"the panes were not complete after {TIMEOUT_S} s; asked {json.dumps(asked)}"]
            await asyncio.sleep(POLL_S)
        times, _ = await page.settle(t0, {"panes complete": ct_key})
        chunks = [path for path in asked if CHUNK_KEY.match(path)]
        print(f"the panes were complete {times['panes complete']} s after the page was asked for the volume, having "
              f"asked for {len(asked)} objects, {len(chunks)} of them chunks, {min(asked.values())} to "
              f"{max(asked.values())} times each")
        failures = []
        if once := [path for path, count in asked.items() if count < 2]:
            failures.append(f"the page asked for {', '.join(once)} once, taking its 503")
        if not chunks:
            failures.append("the page read no chunk")
        return failures
    finally:
        await runner.cleanup()
