import asyncio
import time

from aiohttp import web

from browser import find_free_port
from page import TIMEOUT_S

STATUS = "document.querySelector('#status').title"


async def check(page, out):
    ended = asyncio.Event()

    async def unanswered(request):
        await ended.wait()
        return web.Response(status=503)

    app = web.Application()
    app.router.add_get("/{path:.*}", unanswered)
    runner = web.AppRunner(app)
    await runner.setup()
    port = find_free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    volume = f"http://127.0.0.1:{port}/unanswered.zarr/"
    try:
        t0 = time.monotonic()
        await page.open(f"?volume={volume}")
        limit_s = await page.config("ZARR_METADATA_TIME_LIMIT_MS") / 1000
        status = await page.wait_text(STATUS, lambda status: status.startswith("No answer"), TIMEOUT_S)
        waited = time.monotonic() - t0
        print(f"the status bar said {status!r} {waited:.1f} s after the page was asked for the volume")
        page.expected("No answer for")
        failures = []
        if status != f"No answer for {volume}.zattrs in {limit_s:g} s.":
            failures.append(f"the status bar says {status!r}, not that the volume's .zattrs had no answer")
        if waited < limit_s:
            failures.append(f"the page gave up {waited:.1f} s after it was asked for the volume, before {limit_s:g} s")
        return failures
    finally:
        ended.set()
        await runner.cleanup()
