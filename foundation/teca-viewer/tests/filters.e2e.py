import base64
import json
import time

from page import CENTER_XYZ, NEUTRAL_XY, VOLUME, ct_key, lines_key, surface_key

ZOOM = "4"
PANES = ("xy", "3d")
SIGMA = "4"
THRESHOLDED_PX = [(x, y) for x in (-90, -45, 45, 90) for y in (-70, -35, 35, 70)]
PAN_PX = 160
EDGE_PX = 8
ROWS_CLEAR_PX = 50
EDGE_GREY_LEVELS = 0.5
SURFACE_WHEELS = 30
PIXELS = """(async ([png, points]) => {
  const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
    { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  return points.map(([x, y]) => [...context.getImageData(x, y, 1, 1).data.slice(0, 3)]);
})"""
BANDS = """(async ([pngs, bands, clear]) => {
  const images = [];
  for (const png of pngs) {
    const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
      { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    images.push(context.getImageData(0, 0, bitmap.width, bitmap.height));
  }
  const [a, b] = images;
  return bands.map(([xa, xb, width]) => {
    let sum = 0;
    for (let y = clear; y < a.height - clear; y++)
      for (let x = 0; x < width; x++)
        sum += Math.abs(a.data[4 * (y * a.width + xa + x)] - b.data[4 * (y * b.width + xb + x)]);
    return sum / (width * (a.height - 2 * clear));
  });
})"""


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    if (place := await page.place("#filters-section")) != "data":
        failures.append(f"its card is on {place!r}, not the Data tab")
    await page.reveal("#filters-section")
    t0 = time.monotonic()
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", ZOOM)
    await page.settle(t0, {"panes complete": ct_key})

    async def pictures(name):
        rects = await page.panes()
        hashes = {}
        for pane in PANES:
            png, pixels = await page.picture(rects[pane])
            (out / f"{name}-{pane}.png").write_bytes(png)
            hashes[pane] = pixels["sha256"]
        return hashes

    async def step(action):
        t0 = time.monotonic()
        await action
        await page.settle(t0, {"panes complete": ct_key})

    ct = await pictures("ct")
    await page.choose("#filters-kind", "gaussian")
    await step(page.click("#filters-add"))
    blurred = await pictures("blurred")
    for pane in PANES:
        if blurred[pane] == ct[pane]:
            failures.append(f"the blur left {pane} as it was, {out / f'blurred-{pane}.png'}")
    await step(page.choose("#filters-items .filters-value", SIGMA))
    if (await pictures("wider"))["xy"] == blurred["xy"]:
        failures.append(f"σ {SIGMA} drew the picture of σ 1, {out / 'wider-xy.png'}")

    await page.choose("#filters-kind", "threshold")
    await step(page.click("#filters-add"))
    thresholded = await pictures("thresholded")
    rect = (await page.panes())["xy"]
    png, _ = await page.picture(rect)
    centre = (rect["width"] / 2, rect["height"] / 2)
    points = [[round(centre[0] + dx), round(centre[1] + dy)] for dx, dy in THRESHOLDED_PX]
    for point, rgb in zip(points, await page.evaluate(f"{PIXELS}({json.dumps([base64.b64encode(png).decode(), points])})")):
        if rgb not in ([0, 0, 0], [255, 255, 255]):
            failures.append(f"the thresholded xy pane is {rgb} at {point}, {out / 'thresholded-xy.png'}")

    for index in (1, 2):
        await step(page.choose(f"#filters-items .layer-row:nth-child({index}) input[type=checkbox]", checked=False))
    if await pictures("off") != ct:
        failures.append(f"turning the stages off did not give the CT back, {out / 'off-xy.png'}")
    for index in (1, 2):
        await step(page.choose(f"#filters-items .layer-row:nth-child({index}) input[type=checkbox]", checked=True))

    t0 = time.monotonic()
    await page.open(await page.address())
    await page.settle(t0, {"panes complete": ct_key})
    if await pictures("reloaded") != thresholded:
        failures.append(f"the reloaded page drew another picture, {out / 'reloaded-xy.png'}")

    for _ in (1, 2):
        await step(page.click("#filters-items .layer-row button[title=Remove]"))
    if await pictures("removed") != ct:
        failures.append(f"removing the stages did not give the CT back, {out / 'removed-xy.png'}")

    async def edges(pane, keys):
        rect = (await page.panes())[pane]
        before, _ = await page.picture(rect)
        (out / f"edges-{pane}-before.png").write_bytes(before)
        x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
        await page.mouse("mouseMoved", x, y)
        await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseMoved", x + PAN_PX, y, button="left", buttons=1)
        await page.mouse("mouseReleased", x + PAN_PX, y, button="left", buttons=0, clickCount=1)
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.evaluate("document.activeElement.blur(), true")
        await page.settle(time.monotonic(), keys)
        after, _ = await page.picture(rect)
        (out / f"edges-{pane}-after.png").write_bytes(after)
        width = round(rect["width"])
        bands = [[0, PAN_PX, EDGE_PX], [width - PAN_PX - EDGE_PX, width - EDGE_PX, EDGE_PX]]
        pngs = [base64.b64encode(png).decode() for png in (before, after)]
        greys = await page.evaluate(f"{BANDS}({json.dumps([pngs, bands, ROWS_CLEAR_PX])})")
        return [f"the blurred {pane} pane's {side} edge is {grey:.2f} grey levels off the same place inside it, "
                f"{out / f'edges-{pane}-before.png'}, {out / f'edges-{pane}-after.png'}"
                for side, grey in zip(("left", "right"), greys) if grey > EDGE_GREY_LEVELS]

    await page.choose("#filters-kind", "gaussian")
    await step(page.click("#filters-add"))
    await step(page.choose("#filters-items .filters-value", SIGMA))
    failures += await edges("xy", {"panes complete": ct_key})

    keys = {"surface": surface_key, "panes complete": ct_key, "lines": lines_key}
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    t0 = time.monotonic()
    await page.key("u")
    await page.settle(t0, {"surface": surface_key})
    t0 = time.monotonic()
    await page.choose("#surface-layout-mode", "below")
    await page.settle(t0, keys)
    x, y = await page.pane_center("surface")
    await page.mouse("mouseMoved", x, y)
    for _ in range(SURFACE_WHEELS):
        await page.mouse("mouseWheel", x, y, deltaX=0, deltaY=-240)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(time.monotonic(), keys)
    rect = (await page.panes())["surface"]
    _, blurred = await page.picture(rect)
    t0 = time.monotonic()
    await page.choose("#filters-items input[type=checkbox]", checked=False)
    await page.settle(t0, keys)
    png, sharp = await page.picture(rect)
    (out / "surface-ct.png").write_bytes(png)
    if sharp["sha256"] == blurred["sha256"]:
        failures.append(f"the blur left the Surface as it was, {out / 'surface-ct.png'}")
    t0 = time.monotonic()
    await page.choose("#filters-items input[type=checkbox]", checked=True)
    await page.settle(t0, keys)
    failures += await edges("surface", keys)
    return failures
