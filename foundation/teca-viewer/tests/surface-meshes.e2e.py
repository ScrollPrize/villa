import asyncio
import base64
import json
import re
import time

from page import CENTER_XYZ, NEUTRAL_XY, POLL_S, TIMEOUT_S, VOLUME, ct_key, lines_key

BUILT = re.compile(r"^1 surface, (\d+\.\d\d) M triangles, grid step (\d+)$")
CENTER_ZYX = [float(value) for value in reversed(CENTER_XYZ.split(","))]
REACH = 256
AWAY_PX = 30
AROUND_PX = 3
CHANGED = """(async ([a, b, away, around]) => {
  const rgba = async (png) => {
    const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
      { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    return context.getImageData(0, 0, bitmap.width, bitmap.height);
  };
  const [before, after] = await Promise.all([a, b].map(rgba));
  const { width, height } = before;
  const differs = (x, y) => [0, 1, 2].some((c) => before.data[4 * (y * width + x) + c] !== after.data[4 * (y * width + x) + c]);
  let best = null;
  for (let y = around; y < height - around; y++)
    for (let x = around; x < width - around; x++) {
      const distance = Math.hypot(x - width / 2, y - height / 2);
      if (distance < away || (best !== null && distance >= best[2])) continue;
      let all = true;
      for (let dy = -around; dy <= around && all; dy++) for (let dx = -around; dx <= around && all; dx++) all = differs(x + dx, y + dy);
      if (all) best = [x, y, distance];
    }
  return best === null ? null : best.slice(0, 2);
})"""
GPU = "Object.entries(window.__tecaDebug.rendererStats()).map(([p, r]) => [p, r.textures, r.geometries])"


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    if (place := await page.place("#surface-meshes-section")) != "view":
        failures.append(f"its card is on {place!r}, not the View tab")
    await page.reveal("#surface-meshes-section")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})

    pngs = {}

    async def picture(name):
        png, pixels = await page.picture((await page.panes())["3d"])
        (out / f"{name}.png").write_bytes(png)
        pngs[name] = base64.b64encode(png).decode()
        return pixels["sha256"]

    async def note():
        return await page.evaluate("document.querySelector('#surface-meshes-note').textContent")

    planes = await picture("planes")
    before = await page.evaluate(GPU)
    t0 = time.monotonic()
    await page.choose("#surface-meshes-show", checked=True)
    while (built := BUILT.match(await note())) is None:
        if time.monotonic() - t0 > TIMEOUT_S:
            raise TimeoutError(f"the mesh was not built in {TIMEOUT_S} s: {await note()!r}")
        await asyncio.sleep(POLL_S)
    print(f"the mesh of {built[1]} M triangles, grid step {built[2]}, was built in {time.monotonic() - t0:.2f} s")
    t0 = time.monotonic()
    await page.settle(t0, {"panes complete": ct_key})
    added = await page.evaluate(GPU)
    if await picture("mesh") == planes:
        failures.append(f"showing the mesh left the 3D pane as it was, {out / 'mesh.png'}")

    area = (await page.panes())["3d"]
    at = await page.evaluate(f"{CHANGED}({json.dumps([pngs['planes'], pngs['mesh'], AWAY_PX, AROUND_PX])})")
    if at is None:
        raise RuntimeError(f"the mesh changed no pixel {AWAY_PX} from the 3D pane's centre, {out / 'mesh.png'}")
    x, y = round(area["x"]) + at[0], round(area["y"]) + at[1]
    t0 = time.monotonic()
    await page.mouse("mouseMoved", x, y)
    await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
    await page.mouse("mouseReleased", x, y, button="left", buttons=0, clickCount=1)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(t0, {"panes complete": ct_key})
    picked = (await page.evaluate("window.__tecaDebug.session()"))["center_ZYX"]
    offsets = [abs(a - b) for a, b in zip(picked, CENTER_ZYX)]
    if min(offsets) < 1 or max(offsets) > REACH:
        failures.append(f"a click on the mesh focused {picked}, {offsets} from the focus: on a plane or outside the box")
    t0 = time.monotonic()
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.settle(t0, {"panes complete": ct_key})

    t0 = time.monotonic()
    await page.choose("#surface-meshes-show", checked=False)
    await page.settle(t0, {"panes complete": ct_key})
    if await picture("hidden") != planes:
        failures.append(f"hiding the mesh did not give the 3D pane back, {out / 'hidden.png'}")
    took = [[a[1] - b[1], a[2] - b[2]] for a, b in zip(added, before)]
    freed = [[a[1] - h[1], a[2] - h[2]] for a, h in zip(added, await page.evaluate(GPU))]
    print(f"showing the mesh took {took} textures and geometries per pane")
    if freed != took:
        failures.append(f"hiding the mesh freed {freed} textures and geometries per pane, not the {took} showing it took")
    return failures
