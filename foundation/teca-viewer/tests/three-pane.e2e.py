import json
import time

from page import CENTER_XYZ, NEUTRAL_XY, TIMEOUT_S, VOLUME, ct_key, lines_key, slices_key, surface_key

MOVE_VOXELS = 64
EDGE_PX = 2
DOCK = "!document.querySelector('#surface-layout-mode').disabled"
MOVED_XYZ = ", ".join(f"{float(v) + MOVE_VOXELS:.2f}" for v in CENTER_XYZ.split(","))
ZOOM = "4"
BARS = """(() => Object.fromEntries(["xy", "zy", "xz"].map((name) => {
  const bar = document.querySelector("#viewer-" + name).parentElement.querySelector(".scale-bar");
  return [name, [bar.hidden, bar.dataset.label ?? null, bar.style.getPropertyValue("--scale-bar-width")]];
})))()"""


def drawn_key(s):
    if s is None or s["ct"] is None or s["ct"]["pending"] != 0 or s["ct"]["upload"] != 0 or s["cutting"] != 0:
        return None
    planes = slices_key(s["ct"]) if s["ct"]["panes"] else "none"
    return None if planes is None else json.dumps([planes, s["lines"], s["frames"]])


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", "1")
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")

    async def picture(layout, when):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"drawn": drawn_key})
        r = (await page.panes())["3d"]
        png, pixels = await page.picture({"x": r["x"] + EDGE_PX, "y": r["y"] + EDGE_PX,
                                          "width": r["width"] - 2 * EDGE_PX, "height": r["height"] - 2 * EDGE_PX})
        (out / f"{layout}-{when}.png").write_bytes(png)
        return pixels["sha256"]

    async def move(to):
        await page.type_into("position-number", to)
        await page.key("Enter")
        await page.evaluate("document.activeElement.blur(), true")

    async def scale_bars(when, zoom):
        bars = await page.evaluate(BARS)
        await page.choose("#zoom-number", zoom)
        if bars != (fresh := await page.evaluate(BARS)):
            failures.append(f"{when}: the CT panes' scale bars are {bars}, where a zoom to the same {zoom} sets {fresh}")

    async def shown(layout):
        changed = await picture(layout, "changed")
        await move(MOVED_XYZ)
        moved = await picture(layout, "moved")
        if moved == changed:
            failures.append(f"{layout}: a move of the focus left the 3D pane as it was, {out / f'{layout}-moved.png'}")
        await page.open(await page.address())
        if await picture(layout, "reloaded") != moved:
            failures.append(f"{layout}: a reload drew the 3D pane otherwise than the move, so a plane it shows was not "
                            f"drawn at the focus, {out / f'{layout}-moved.png'} against {out / f'{layout}-reloaded.png'}")
        await move(CENTER_XYZ)
        if await picture(layout, "back") != changed:
            failures.append(f"{layout}: back at the focus, the reloaded page drew the 3D pane otherwise than the layout "
                            f"change, {out / f'{layout}-changed.png'} against {out / f'{layout}-back.png'}")

    await page.key("4")
    await page.key("z")
    await shown("grid-3d-maximised")
    await page.click('#view-strip [data-view="grid"]')
    t0 = time.monotonic()
    await page.click('#view-strip [data-view="surface"]')
    await page.settle(t0, {"surface": surface_key})
    await page.choose("#zoom-number", ZOOM)
    await page.click('#view-strip [data-view="grid"]')
    await scale_bars("the Surface closed after a zoom while alone", ZOOM)
    await page.choose("#zoom-number", "1")
    t0 = time.monotonic()
    await page.click('#view-strip [data-view="surface"]')
    await page.settle(t0, {"surface": surface_key})
    for layout, steps in (("below-xy-3d", (("#surface-layout-mode", "below", None),
                                           ("[data-surface-plane='3d']", None, True))),
                          ("right-3d", (("#surface-layout-mode", "right", None),
                                        ("[data-surface-plane='xy']", None, False)))):
        for selector, value, checked in steps:
            await page.choose(selector, value, checked)
            await page.browser.wait_for(DOCK, TIMEOUT_S)
        await page.settle(time.monotonic(), {"surface": surface_key})
        if layout == "below-xy-3d":
            await page.key("1")
            await page.key("z")
            await scale_bars("XY maximised in the dock", "1")
            await page.key("z")
        await shown(layout)
    await page.key("4")
    await page.key("z")
    await shown("dock-3d-maximised")
    return failures
