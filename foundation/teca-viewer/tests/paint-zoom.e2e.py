import asyncio
import gzip
import math
import time
import zipfile

from page import CENTER_XYZ, TIMEOUT_S, VOLUME, ct_key, lines_key, surface_key

WHEELS = 20
ZOOM_OUTS = [26, 14, 4]
NOTCH = 1.12
RADIUS = "32"
ERASE_RADIUS = "64"
STROKE_PX = 120
STEP_PX = 10
PAD_PX = 1
PACKED_S = 1.0
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-labels.zarr.zip"
READY = "!document.querySelector('#paint-tool').disabled"


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

    async def wheel(notches, delta_y):
        await page.mouse("mouseMoved", x, y)
        for _ in range(notches):
            await page.mouse("mouseWheel", x, y, deltaX=0, deltaY=delta_y)

    async def picture(name):
        await page.mouse("mouseMoved", 150, 950)
        await page.settle(time.monotonic(), {"surface": surface_key})
        png, pixels = await page.picture(area)
        (out / f"{name}.png").write_bytes(png)
        return pixels["sha256"]

    async def stroke(radius, left, right):
        await page.choose("#paint-radius", radius)
        await page.mouse("mouseMoved", left, y)
        await page.mouse("mousePressed", left, y, button="left", buttons=1, clickCount=1)
        for step_x in [*range(left + STEP_PX, right, STEP_PX), right]:
            await page.mouse("mouseMoved", step_x, y, button="left", buttons=1)
        await page.mouse("mouseReleased", right, y, button="left", buttons=0, clickCount=1)
        await asyncio.sleep(PACKED_S)

    async def labelled(name):
        path = out / f"{name}.zip"
        (await page.download("#paint-save", SAVED)).rename(path)
        with zipfile.ZipFile(path) as saved:
            chunks = [entry for entry in saved.namelist() if entry.startswith("0/") and not entry.startswith("0/.")]
            return sum(len(data) - data.count(0) for data in (gzip.decompress(saved.read(entry)) for entry in chunks))

    await wheel(WHEELS, -240)
    before = await picture("before")
    await page.click("#paint-tool")
    for notches in ZOOM_OUTS:
        await page.choose("#paint-erase", checked=False)
        await stroke(RADIUS, x - STROKE_PX // 2, x + STROKE_PX // 2)
        if await picture(f"painted-{notches}") == before:
            failures.append(f"the stroke before erasing {notches} notches out left the Surface as it was, "
                            f"{out / f'painted-{notches}.png'}")
        painted = await labelled(f"painted-{notches}")
        if painted == 0:
            failures.append(f"the stroke before erasing {notches} notches out set no voxel")
        await wheel(notches, 240)
        await picture(f"zoomed-out-{notches}")
        await page.choose("#paint-erase", checked=True)
        half = STROKE_PX / 2 / NOTCH**notches
        await stroke(ERASE_RADIUS, math.floor(x - half) - PAD_PX, math.ceil(x + half) + PAD_PX)
        left = await labelled(f"erased-{notches}")
        if left:
            failures.append(f"erasing {notches} notches out along the stroke's cells left {left} of its {painted} "
                            f"voxels, {out / f'erased-{notches}.zip'}")
        await wheel(notches, -240)
    back = await picture("zoomed-back")
    await page.open(await page.address())
    await page.browser.wait_for(READY, TIMEOUT_S)
    if await picture("reloaded") != back:
        failures.append(f"zoomed back in after erasing {ZOOM_OUTS[-1]} notches out, the Surface draws other pixels than "
                        f"after a reload, {out / 'zoomed-back.png'} and {out / 'reloaded.png'}")
    return failures
