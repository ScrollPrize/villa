import asyncio
import base64
import json
import math
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, NEUTRAL_XY, POLL_S, SURFACE, TIMEOUT_S, VOLUME, ct_key, lines_key, surface_key

ZOOM = 4
STEP_PX = 10
SCRIBBLE_PX = (-120, 120)
ROW_PX = 40
AWAY_PX = 150
TURNS = 30
# Steps of the wheel in from the Surface's opening zoom to 0.44 full pixels a pixel, where nodes 5 voxels apart stand
# 11 px apart, past a disc.
WHEELS = 36
RING_PX = (2, 3, 4)
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser."
NOTE = "document.querySelector('#points-note').textContent"
PIXELS = """(async ([png, points]) => {
  const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
    { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  return points.map(([x, y]) => [...context.getImageData(x, y, 1, 1).data.slice(0, 3)]);
})"""
CELLS = """(async ([url, points_ZYX]) => {
  const { SurfaceCells } = await import("./src/extensions/points/surface.js");
  const cells = new SurfaceCells(await window.__viewer.readSurface(url, new AbortController().signal));
  return points_ZYX.map((point_ZYX) => cells.nearest(point_ZYX, 64));
})"""


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.reveal("#points-section")
    if (depth := await page.evaluate("document.querySelector('#points-depth').value")) != "0":
        failures.append(f"Depth reads {depth} at load, not 0")
        await page.choose("#points-depth", "0")
    await page.choose("#points-follow", checked=False)
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.evaluate("document.activeElement?.blur(), true")
    await page.choose("#zoom-number", str(ZOOM))
    await page.settle(t0, {"panes complete": ct_key})
    rect = (await page.panes())["xy"]
    x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
    voxels_per_px = await page.config("BASE_VIEW_WIDTH_VOXELS") / ZOOM / min(rect["width"], rect["height"])
    await page.click("#points-tool")

    async def saved():
        path = await page.download("#points-save", SAVED)
        with zipfile.ZipFile(path) as zipped:
            root = ElementTree.fromstring(zipped.read(FINEST))
        path.unlink()
        nodes = sorted([int(node.get("id")), *[float(node.get(axis)) for axis in "xyz"]] for node in root.iter("node"))
        active = root.find("parameters/activeNode")
        return nodes, None if active is None else int(active.get("id"))

    async def scribble(row, from_x, to_x, before):
        await page.mouse("mouseMoved", from_x, row)
        await page.mouse("mousePressed", from_x, row, button="left", buttons=1, clickCount=1)
        for step_x in range(from_x + STEP_PX, to_x + 1, STEP_PX):
            await page.mouse("mouseMoved", step_x, row, button="left", buttons=1)
            await asyncio.sleep(0.01)
        await page.mouse("mouseReleased", to_x, row, button="left", buttons=0, clickCount=1)
        await page.wait_text(NOTE, lambda note: note.endswith(KEPT), 60)
        known = {node[0] for node in before}
        return [node for node in (await saved())[0] if node[0] not in known]

    async def discs(name, places, area, keys):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), keys)
        png, _ = await page.picture(area)
        (out / f"{name}.png").write_bytes(png)
        offsets = [0, *[-dy for dy in RING_PX], *RING_PX]
        at = [[round(px - area["x"]), round(py - area["y"]) + dy] for px, py in places for dy in offsets]
        rgbs = await page.evaluate(f"{PIXELS}({json.dumps([base64.b64encode(png).decode(), at])})")
        drawn = []
        for index in range(len(places)):
            middle, *ring = rgbs[len(offsets) * index:len(offsets) * (index + 1)]
            coloured = max(middle) - min(middle) > 60 or min(middle) >= 230
            above, below = ring[:len(RING_PX)], ring[len(RING_PX):]
            drawn.append(coloured and all(any(max(rgb) < 40 for rgb in side) for side in (above, below)))
        return drawn

    async def taken(name, placed, places):
        count = len((await saved())[0])
        middle = len(placed) // 2
        at_x, at_y = places[middle]
        await page.mouse("mouseMoved", at_x, at_y)
        await page.mouse("mousePressed", at_x, at_y, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", at_x, at_y, button="left", buttons=0, clickCount=1)
        nodes, active = await saved()
        if len(nodes) != count or active != placed[middle][0]:
            failures.append(f"a click on {name}'s node {placed[middle][0]} left {len(nodes)} nodes, not {count}, and "
                            f"node {active} selected")

    keys = {"panes complete": ct_key}
    before = []
    for name, row in [("xy", y - ROW_PX), ("xy turned 30°", y + ROW_PX)]:
        if name != "xy":
            await page.mouse("mouseMoved", x, y + AWAY_PX)
            for _ in range(TURNS):
                await page.key("ArrowRight")
            await page.settle(time.monotonic(), keys)
        tag = name.replace(" ", "-").replace("°", "")
        placed = await scribble(row, x + SCRIBBLE_PX[0], x + SCRIBBLE_PX[1], before)
        places = [(x + SCRIBBLE_PX[0] + math.dist(node[1:], placed[0][1:]) / voxels_per_px, row) for node in placed]
        drawn = await discs(tag, places, rect, keys)
        if not all(drawn):
            failures.append(f"{name} draws {sum(drawn)} of the {len(drawn)} nodes of the scribble drawn on it, "
                            f"{out / f'{tag}.png'}")
        for key, then in [("F", "a slice on"), ("D", "back")]:
            await page.mouse("mouseMoved", x, y + AWAY_PX)
            await page.key(key, shift=True)
            stepped = await discs(f"{tag}-{key}", places, rect, keys)
            if stepped != [key == "D"] * len(placed):
                failures.append(f"{name} {then} after Shift+{key} draws {sum(stepped)} of the scribble's {len(placed)} "
                                f"nodes, not {'all' if key == 'D' else 'none'}, {out / f'{tag}-{key}.png'}")
        await taken(name, placed, places)
        before = (await saved())[0]

    await page.click("#points-clear")
    t0 = time.monotonic()
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.evaluate("document.activeElement?.blur(), true")
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement?.blur(), true")
    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    area = (await page.panes())["surface"]
    x, y = round(area["x"] + area["width"] / 2), round(area["y"] + area["height"] / 2)
    for _ in range(WHEELS):
        await page.mouse("mouseWheel", x, y, deltaX=0, deltaY=-100)
        await asyncio.sleep(0.05)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(time.monotonic(), {"surface": surface_key})
    await page.click("#points-tool")
    row, from_x, to_x = y + ROW_PX, x + SCRIBBLE_PX[0], x + SCRIBBLE_PX[1]
    placed = await scribble(row, from_x, to_x, [])
    await page.extension_viewer()
    cells = await page.evaluate(f"{CELLS}({json.dumps([SURFACE, [node[3:0:-1] for node in placed]])})")
    view = await page.evaluate("window.__viewer.surfaceView()")
    if not placed or None in cells:
        failures.append(f"the Surface's scribble placed {len(placed)} nodes, with cells {cells}")
        return failures
    rows_a_px, columns_a_px = [(view["stop_HW"][axis] - view["start_HW"][axis]) / area[side]
                               for axis, side in [(0, "height"), (1, "width")]]
    places = [(from_x + (column - cells[0][1]) / columns_a_px, row + (cell_row - cells[0][0]) / rows_a_px)
              for cell_row, column in cells]
    if abs(places[-1][0] - to_x) > 2 * STEP_PX:
        failures.append(f"the Surface's scribble from x {from_x} to {to_x} has its last node at x {places[-1][0]:.1f}")
        return failures
    t0 = time.monotonic()
    while not all(drawn := await discs("surface", places, area, {"surface": surface_key})):
        if time.monotonic() - t0 > TIMEOUT_S:
            failures.append(f"the Surface draws {sum(drawn)} of the {len(drawn)} nodes of the scribble drawn on it, "
                            f"{out / 'surface.png'}")
            break
        await asyncio.sleep(POLL_S)
    await taken("the Surface", placed, places)
    return failures
