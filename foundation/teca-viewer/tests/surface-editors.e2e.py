import asyncio
import json
import time

from page import BLANK_COLORS, NEUTRAL_XY, VOLUME, ct_key, lines_key, surface_key

TRANSFER_POINT_LIMIT = 8
PANEL = """(() => ({
  points: document.querySelector('#surface-volumetric-transfer-points').children.length,
  addDisabled: document.querySelector('#surface-volumetric-transfer-add').disabled,
  secondOpacity: Number(document.querySelector('.surface-transfer-point:nth-child(2) input[type=range]')?.value),
  preset: document.querySelector('#surface-volumetric-transfer-preset').value,
  maskEnabled: document.querySelector('#surface-relief-mask-enabled').checked,
  maskStatus: document.querySelector('#surface-relief-mask-status').textContent,
  overlayHidden: document.querySelector('.surface-relief-overlay').hasAttribute('hidden'),
  handles: document.querySelectorAll('.surface-relief-control').length,
  cursor: document.querySelector('#cursor-value').textContent,
}))()"""
HANDLE = """(() => {
  const handle = [...document.querySelectorAll('.surface-relief-control')].find((e) => {
    const r = e.getBoundingClientRect();
    return r.y > 100 && document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2) === e;
  });
  handle.id = 'dragged-handle';
  const r = handle.getBoundingClientRect();
  return [r.x + r.width / 2, r.y + r.height / 2, handle.getAttribute('aria-label')];
})()"""
DRAG_STEPS = 5
DRAG_STEP_XY = (8, -12)


async def check(page, out):
    failures = []

    async def edited(label, expected):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"surface": surface_key})
        panel = await page.evaluate(PANEL)
        wrong = {key: panel[key] for key, value in expected.items() if panel[key] != value}
        if wrong:
            failures.append(f"{label}: the panel shows {wrong}, not {json.dumps({key: expected[key] for key in wrong})}")
        return panel

    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")
    await page.click("[data-ltab=view]")
    await page.choose("#workspace-view", "volume")
    points = (await edited("volume display", {}))["points"]

    await page.click("#surface-volumetric-transfer-add")
    await page.click("#surface-volumetric-transfer-add")
    await edited("transfer points added", {"points": points + 2})
    await page.choose("#surface-volumetric-transfer-points .surface-transfer-point:nth-child(2) input[type=range]", "0.3")
    await edited("transfer point moved", {"points": points + 2, "secondOpacity": 0.3})
    await page.click("#surface-volumetric-transfer-points .surface-transfer-point:nth-child(3) button")
    await edited("another transfer point removed", {"points": points + 1, "secondOpacity": 0.3})
    await page.click("#surface-volumetric-transfer-points .surface-transfer-point:nth-child(2) button")
    await edited("transfer point removed", {"points": points})
    for _ in range(TRANSFER_POINT_LIMIT - points):
        await page.click("#surface-volumetric-transfer-add")
    await edited("transfer curve full", {"points": TRANSFER_POINT_LIMIT, "addDisabled": True})
    await page.choose("#surface-volumetric-transfer-preset", "ink")
    await edited("ink preset", {"preset": "ink"})

    await page.choose("#surface-relief-mask-enabled", checked=True)
    handles = (await edited("relief mask on", {"maskEnabled": True, "overlayHidden": False}))["handles"]
    await page.choose("#surface-relief-mask-boundary", "bottom")
    await page.choose("#surface-relief-mask-depth-number", "0.7")
    await page.click("#surface-relief-mask-set")
    status = (await edited("relief control set", {"handles": handles + 3}))["maskStatus"]
    if not status.startswith("bottom control set at U "):
        failures.append(f"setting a relief control says {status!r}, not where it was set")

    x, y, label = await page.evaluate(HANDLE)
    await page.mouse("mouseMoved", x, y)
    await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
    for step in range(1, DRAG_STEPS + 1):
        await page.mouse("mouseMoved", x + step * DRAG_STEP_XY[0], y + step * DRAG_STEP_XY[1], button="left", buttons=1)
        await asyncio.sleep(0.05)
    await page.mouse("mouseReleased", x + DRAG_STEPS * DRAG_STEP_XY[0], y + DRAG_STEPS * DRAG_STEP_XY[1], button="left", buttons=0, clickCount=1)
    await edited("relief control dragged", {"handles": handles + 3})
    dragged = await page.evaluate("document.querySelector('#dragged-handle').getAttribute('aria-label')")
    if dragged == label:
        failures.append(f"dragging the relief control left it at {label}")

    await page.choose("#surface-relief-mask-overlay-visible", checked=False)
    await edited("relief overlay hidden", {"overlayHidden": True})
    await page.choose("#surface-relief-mask-overlay-visible", checked=True)
    await page.click("#surface-relief-mask-remove")
    await edited("relief control removed", {"handles": handles, "maskStatus": "Nearest bottom control removed."})
    await page.click("#surface-relief-mask-reset")
    await edited("relief mask reset", {"maskEnabled": False, "maskStatus": "Top 2 ridges, bottom 2 ridges."})

    await page.mouse("mouseMoved", *await page.pane_center("surface"))
    await asyncio.sleep(0.5)
    cursor = (await page.evaluate(PANEL))["cursor"]
    if not cursor.startswith("surface XYZ"):
        failures.append(f"hovering the Surface reads {cursor!r}, not a surface point")
    png, pixels = await page.picture((await page.panes())["surface"])
    (out / "surface.png").write_bytes(png)
    if pixels["centralColors"] < BLANK_COLORS:
        failures.append(f"the Surface slab is blank after the edits: {pixels['centralColors']} colors, {out / 'surface.png'}")
    return failures
