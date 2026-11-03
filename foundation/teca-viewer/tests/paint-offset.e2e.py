import asyncio
import gzip
import json
import time
import zipfile

from page import CENTER_XYZ, TIMEOUT_S, VOLUME, ct_key, lines_key, surface_key

ZOOM_INS = 6
RADIUS = "256"
MOVES = 12
MOVE_PX = 20
OFFSET = "8"
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-labels.zarr.zip"
READY = "!document.querySelector('#paint-tool').disabled"
NOTE = "document.querySelector('#paint-note').textContent"
KEPT = ", kept in this browser."


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.reveal("#paint-section")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", "1")
    await page.settle(t0, {"panes complete": ct_key})
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    area = (await page.panes())["surface"]
    x, y = round(area["x"] + area["width"] / 2), round(area["y"] + area["height"] / 2)
    await page.mouse("mouseMoved", x, y)
    for _ in range(ZOOM_INS):
        await page.mouse("mouseWheel", x, y, deltaX=0, deltaY=-240)
    await page.settle(time.monotonic(), {"surface": surface_key})
    await page.choose("#paint-radius", RADIUS)
    await page.click("#paint-tool")

    async def stroke(offset):
        left = x - MOVES * MOVE_PX // 2
        await page.mouse("mouseMoved", left, y)
        await page.mouse("mousePressed", left, y, button="left", buttons=1, clickCount=1)
        for move in range(1, MOVES + 1):
            await page.mouse("mouseMoved", left + move * MOVE_PX, y, button="left", buttons=1)
        await page.mouse("mouseReleased", left + MOVES * MOVE_PX, y, button="left", buttons=0, clickCount=1)
        if offset is not None:
            await page.choose("#surface-offset-number", offset)
        deadline = time.monotonic() + TIMEOUT_S
        while not (note := await page.evaluate(NOTE)).endswith(KEPT) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return note

    async def labelled(name):
        path = out / f"{name}.zip"
        (await page.download("#paint-save", SAVED)).rename(path)
        with zipfile.ZipFile(path) as saved:
            chunks = {entry: gzip.decompress(saved.read(entry)) for entry in saved.namelist()
                      if entry.startswith("0/") and not entry.startswith("0/.")}
        return {entry: data for entry, data in chunks.items() if data.count(0) < len(data)}

    if not (note := await stroke(None)).endswith(KEPT):
        return [f"the stroke says {note!r}, not kept"]
    kept = await labelled("kept")
    await page.click("#paint-clear")
    await page.browser.wait_for(READY, TIMEOUT_S)
    await page.click("#paint-tool")
    if not (note := await stroke(OFFSET)).endswith(KEPT):
        return [f"the stroke with the offset changed after its release says {note!r}, not kept"]
    changed = await labelled("changed")
    print(f"the stroke labels {len(kept)} finest chunks, and with the offset changed after its release {len(changed)}")
    if not kept:
        failures.append("the stroke set no voxel")
    differing = sorted(set(kept) ^ set(changed) | {entry for entry in kept if changed.get(entry, kept[entry]) != kept[entry]})
    if differing:
        failures.append(f"with the offset changed after its release, the stroke labels other voxels in {len(differing)} "
                        f"of its {len(kept)} finest chunks, {json.dumps(differing[:4])}, {out / 'kept.zip'} and "
                        f"{out / 'changed.zip'}")
    return failures
