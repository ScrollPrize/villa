import asyncio
import re
import time

from page import BLANK_COLORS, NEUTRAL_XY, SURFACE, VOLUME, ct_key, lines_key, surface_key

METADATA = re.compile(r"/(zarr\.json|\.zattrs|\.zarray|\.zgroup)$")
SLICE_STEPS = 5
HOLE_UV, HOLE_ZOOM = [40, 40], 64
GPU = "Object.entries(window.__tecaDebug.rendererStats()).map(([p, r]) => [p, r.textures, r.geometries])"


def metadata_reads(page):
    return sum(1 for r in page.browser.requests.values() if METADATA.search(r["url"]))


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}&overlay={SURFACE}")
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    if "overlay=" in await page.evaluate("location.search"):
        failures.append("the address keeps the surface's link once the session holds the surface")

    await page.evaluate("document.activeElement.blur(), true")
    t0 = time.monotonic()
    await page.key("u")
    await page.settle(t0, {"surface": surface_key})
    reads = metadata_reads(page)
    opened = await page.evaluate(GPU)
    t0 = time.monotonic()
    await page.key("u")
    await page.settle(t0, {"closed": lambda s: "closed" if s is not None and s["surface"] is None else None})
    t0 = time.monotonic()
    await page.key("u")
    await page.settle(t0, {"surface": surface_key})
    if metadata_reads(page) != reads:
        failures.append(f"reopening the Surface read the volume's metadata again: {metadata_reads(page) - reads} reads")
    reopened = await page.evaluate(GPU)
    if reopened != opened:
        failures.append(f"the reopened Surface holds GPU textures and geometries {reopened}, not {opened} as first opened")

    t0 = time.monotonic()
    await page.choose("#surface-layout-mode", "below")
    await page.settle(t0, {"surface": surface_key, "panes complete": ct_key})
    panes = await page.panes()
    if not {"surface", "xy"} <= panes.keys():
        failures.append(f"docking below shows {sorted(panes)}, not the Surface and XY")
        return failures

    position = await page.evaluate("document.querySelector('#position-number').value")
    await page.mouse("mouseMoved", *await page.pane_center("xy"))
    await asyncio.sleep(0.2)
    t0 = time.monotonic()
    for _ in range(SLICE_STEPS):
        await page.key("f")
        await asyncio.sleep(0.1)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(t0, {"surface": surface_key, "panes complete": ct_key})
    stepped = await page.evaluate("document.querySelector('#position-number').value")
    if stepped == position:
        failures.append(f"{SLICE_STEPS} slice steps over XY left the focus at {position}")

    pictures = {}
    for display in ["surface", "volume", "both"]:
        t0 = time.monotonic()
        await page.choose("#workspace-view", display)
        await page.settle(t0, {"surface": surface_key})
        png, pixels = await page.picture((await page.panes())["surface"])
        (out / f"{display}.png").write_bytes(png)
        if pixels["centralColors"] < BLANK_COLORS:
            failures.append(f"the {display} display is blank: {pixels['centralColors']} colors in its central half, {out / display}.png")
        pictures[display] = pixels["sha256"]
    if len(set(pictures.values())) != len(pictures):
        failures.append(f"two displays drew the same picture: {pictures}")

    session = await page.evaluate("window.__tecaDebug.session()")
    session["surfaceView"]["center_UV"] = HOLE_UV
    session["surfaceView"]["zoom"] = HOLE_ZOOM
    t0 = time.monotonic()
    await page.open_session(session)
    await page.settle(t0, {"surface": surface_key})
    center_UV = await page.evaluate("window.__tecaDebug.session().surfaceView.center_UV")
    if center_UV != HOLE_UV:
        failures.append(f"a Surface centred on a hole at {HOLE_UV} moved to {center_UV}")
    return failures
