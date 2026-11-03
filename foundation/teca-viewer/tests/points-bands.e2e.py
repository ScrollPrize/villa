import asyncio
import base64
import json
import time

from page import CENTER_XYZ, NEUTRAL_XY, SURFACE, VOLUME, ct_key, lines_key, surface_key

BAND_VOXELS = 8
BAND_WIDTH = (4, 32)
TOLERANCE_PX = 1.5
ZOOMS = [0.5, 1, 4]
KEPT = ", kept in this browser"
GAP_PX = 20
LENGTH_PX = 240
MOVES = 24
COLUMNS = 15
KOLLESIS = "(r > 200 && g > 130 && g < 230 && b < 90)"
GOLD = """(async ([png, columns]) => {
  const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
    { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  return columns.map(([x, top, bottom]) => {
    const rgba = context.getImageData(x, top, 1, bottom - top + 1).data;
    const gold = [];
    for (let row = 0; row <= bottom - top; row++) {
      const [r, g, b] = rgba.slice(4 * row, 4 * row + 3);
      if (%s) gold.push(top + row);
    }
    return gold.length === 0 ? null : [gold[0], gold[gold.length - 1]];
  });
})""" % KOLLESIS


async def drag(page, x0, y, x1):
    await page.mouse("mouseMoved", x0, y)
    await page.mouse("mousePressed", x0, y, button="left", buttons=1, clickCount=1)
    for step in range(1, MOVES + 1):
        await page.mouse("mouseMoved", x0 + (x1 - x0) * step / MOVES, y, button="left", buttons=1)
        await asyncio.sleep(0.02)
    await page.mouse("mouseReleased", x1, y, button="left", buttons=0, clickCount=1)


async def strokes(page, rect):
    x, y = rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2
    for offset in [-GAP_PX, 0, GAP_PX]:
        await drag(page, x - LENGTH_PX / 2, y + offset, x + LENGTH_PX / 2)
        await page.key("n")
    await page.mouse("mouseMoved", *NEUTRAL_XY)


async def band(page, rect, scale, out, name):
    png, _ = await page.picture(rect)
    (out / name).write_bytes(png)
    x, y = round(rect["width"] / 2), round(rect["height"] / 2)
    reach = round(GAP_PX * scale / 2) - 1
    half = min(LENGTH_PX * scale, rect["width"]) / 4
    columns = [[round(x - half + 2 * half * i / (COLUMNS - 1)), y - reach, y + reach] for i in range(COLUMNS)]
    widths = sorted(0 if ends is None else ends[1] - ends[0] + 1
                    for ends in await page.evaluate(f"{GOLD}({json.dumps([base64.b64encode(png).decode(), columns])})"))
    return widths[COLUMNS // 2]


def wanted(pixels_per_voxel):
    return min(max(BAND_VOXELS * pixels_per_voxel, BAND_WIDTH[0]), BAND_WIDTH[1])


async def pen(page):
    await page.reveal("#points-section")
    await page.evaluate("document.activeElement?.blur(), true")
    await page.browser.wait_for("!document.querySelector('#points-tool').disabled", 60)
    if not await page.evaluate("document.querySelector('#points-tool').classList.contains('active')"):
        await page.key("6")
    if await page.evaluate("document.querySelector('#points-follow').checked"):
        await page.key("O", shift=True)
    if await page.evaluate("document.querySelector('#points-mark-kollesis').getAttribute('aria-pressed')") != "true":
        await page.key("K", shift=True)


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}&overlay={SURFACE}")
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.evaluate("document.activeElement?.blur(), true")
    await page.key("1")
    await page.key("z")
    await page.choose("#zoom-number", "1")
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    await pen(page)
    base_voxels = await page.config("BASE_VIEW_WIDTH_VOXELS")
    rect = (await page.panes())["xy"]
    await strokes(page, rect)
    measured = []
    for zoom in ZOOMS:
        await page.choose("#zoom-number", str(zoom))
        await page.settle(time.monotonic(), {"panes complete": ct_key})
        want = wanted(zoom * min(rect["width"], rect["height"]) / base_voxels)
        measured.append((got := await band(page, rect, zoom, out, f"xy-{zoom}.png"), want))
        if abs(got - want) > TOLERANCE_PX:
            failures.append(f"at zoom {zoom} the XY pane's band is {got} px across, not {want:.1f}, {out / f'xy-{zoom}.png'}")
    print(f"the XY pane's band at zooms {ZOOMS}: {', '.join(f'{got} px, {want:.1f} wanted' for got, want in measured)}")

    await page.click("#points-clear")

    await page.choose("#zoom-number", "1")
    await page.key("z")
    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    await page.extension_viewer()
    per_zoom = (await page.evaluate("window.__viewer.surfaceView().pixel")) * (await page.address_session())["surfaceView"]["zoom"]
    rect = (await page.panes())["surface"]
    at = {zoom: per_zoom * zoom * min(rect["width"], rect["height"]) / base_voxels for zoom in ZOOMS}

    async def surface_at(zoom, restored=True):
        opened = await page.address_session()
        opened["surfaceView"]["zoom"] = at[zoom]
        t0 = time.monotonic()
        await page.open_session(opened)
        await page.settle(t0, {"surface": surface_key})
        if restored:
            await page.wait_text("document.querySelector('#points-note').textContent", lambda note: note.startswith("Brought back"), 60)
            await asyncio.sleep(1)
        await page.settle(time.monotonic(), {"surface": surface_key})
        await page.extension_viewer()
        return 1 / await page.evaluate("window.__viewer.surfaceView().pixel")

    await surface_at(1, restored=False)
    await pen(page)
    await strokes(page, rect)
    await page.wait_text("document.querySelector('#points-note').textContent", lambda note: KEPT in note, 60)
    measured = []
    for zoom in ZOOMS:
        want = wanted(await surface_at(zoom))
        measured.append((got := await band(page, rect, zoom, out, f"surface-{zoom}.png"), want))
        if abs(got - want) > TOLERANCE_PX:
            failures.append(f"at the zoom matching {zoom} the Surface's band is {got} px across, not {want:.1f}, "
                            f"{out / f'surface-{zoom}.png'}")
    print(f"the Surface's band at the zooms matching those: "
          f"{', '.join(f'{got} px, {want:.1f} wanted' for got, want in measured)}")
    return failures
