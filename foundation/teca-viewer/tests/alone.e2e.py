import base64
import json
import time

from page import CENTER_XYZ, NEUTRAL_XY, VOLUME, ct_key, lines_key, slices_key, surface_key

LAYER = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/"
         "ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/")
ADD_LAYER = """(async (url) => {
  const [slices, surface] = await Promise.all([0, 1].map(() => __viewer.openVolume(url)));
  __viewer.addLayer({ slices, surface }, { color_RGB: [0, 1, 0], opacity: 0.65, range: [0, 255] });
  return true;
})"""
PANES = ("xy", "surface")
STATUS = "document.querySelector('#status').textContent"
GPU = "Object.entries(window.__tecaDebug.rendererStats()).map(([p, r]) => [p, r.programs, r.textures, r.geometries])"
DRAWN = {"frames": lambda s: None if s is None else s["frames"]}
EDGE_PX = 2
LEVELS = 3
MATCHED = 0.99
CT_SHARE = 0.25
LAYERS_ALONE = """(async ([ct, both, layers, levels]) => {
  const rgba = async (text) => {
    const blob = await (await fetch("data:image/png;base64," + text)).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    return { width: bitmap.width, height: bitmap.height, data: context.getImageData(0, 0, bitmap.width, bitmap.height).data };
  };
  const [c, b, l] = await Promise.all([ct, both, layers].map(rgba));
  let pixels = 0, grey = 0, matched = 0, green = 0;
  for (let y = c.height >> 2; y < (3 * c.height) >> 2; y++)
    for (let x = c.width >> 2; x < (3 * c.width) >> 2; x++) {
      const i = 4 * (y * c.width + x);
      pixels += 1;
      if (c.data[i] !== c.data[i + 1] || c.data[i] !== c.data[i + 2]) continue;
      grey += 1;
      const added = b.data[i + 1] - b.data[i];
      if (l.data[i] <= levels && l.data[i + 2] <= levels && Math.abs(l.data[i + 1] - added) <= levels) matched += 1;
      if (l.data[i + 1] > levels) green += 1;
    }
  return { pixels, grey, matched, green };
})"""


def layer_key(s):
    return None if s is None or len(s["layers"]) != 1 else slices_key(s["layers"][0])


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.extension_viewer()
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", "1")
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    t0 = time.monotonic()
    await page.key("u")
    await page.settle(t0, {"surface": surface_key})
    t0 = time.monotonic()
    await page.choose("#surface-layout-mode", "below")
    await page.settle(t0, {"panes complete": ct_key, "surface": surface_key})
    rects = {pane: {"x": r["x"] + EDGE_PX, "y": r["y"] + EDGE_PX, "width": r["width"] - 2 * EDGE_PX,
                    "height": r["height"] - 2 * EDGE_PX} for pane, r in (await page.panes()).items()}

    async def pictures(name, panes=PANES):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"panes complete": ct_key, "surface": surface_key})
        shot = {}
        for pane in panes:
            png, pixels = await page.picture(rects[pane])
            (out / f"{name}-{pane}.png").write_bytes(png)
            shot[pane] = (png, pixels["sha256"])
        return shot

    async def press(name, said, ctrl=False):
        gpu = await page.evaluate(GPU)
        pressed = time.monotonic()
        await page.key(" ", ctrl=ctrl)
        if not (status := await page.evaluate(STATUS)).startswith(said):
            failures.append(f"after {name} the status bar says {status!r}, not {said!r}")
        shot = await pictures(name)
        reads = [r["url"] for r in page.browser.requests.values() if r["start"] >= pressed]
        if reads:
            failures.append(f"{name} read {len(reads)} files, as {reads[0]}")
        if (after := await page.evaluate(GPU)) != gpu:
            failures.append(f"{name} changed the panes' GPU programs, textures and geometries from {gpu} to {after}")
        return shot

    def same(shot, name, reference, drew):
        for pane in PANES:
            if shot[pane][1] != reference[pane][1]:
                failures.append(f"{name} did not draw {pane} as {drew}, {out / f'{name}-{pane}.png'}")

    ct = await pictures("ct")
    same(await press("nothing-over", "No layer or image", ctrl=True), "nothing-over", ct, "the CT")
    t0 = time.monotonic()
    await page.evaluate(f"{ADD_LAYER}({json.dumps(LAYER)})")
    await page.settle(t0, {"panes complete": ct_key, "layer": layer_key, "surface": surface_key})
    both = await pictures("both")
    for pane in PANES:
        if both[pane][1] == ct[pane][1]:
            failures.append(f"the layer left {pane} as it was, {out / f'both-{pane}.png'}")

    same(await press("ct-alone", "The CT alone"), "ct-alone", ct, "before the layer was added")
    same(await press("both-again", "The CT with its layers"), "both-again", both, "both")
    layers = await press("layers-alone", "The layers and images alone", ctrl=True)
    for pane in PANES:
        arguments = [base64.b64encode(shot[pane][0]).decode() for shot in (ct, both, layers)] + [LEVELS]
        counts = await page.evaluate(f"{LAYERS_ALONE}({json.dumps(arguments)})")
        print(f"the layer alone on {pane}: {counts}")
        if counts["grey"] < CT_SHARE * counts["pixels"] or counts["green"] == 0:
            failures.append(f"on {pane} the CT draws {counts['grey']} of {counts['pixels']} central pixels grey and the "
                            f"layer alone {counts['green']} green: too few to tell, {out / f'layers-alone-{pane}.png'}")
        elif counts["matched"] < MATCHED * counts["grey"]:
            failures.append(f"the layer alone on {pane} draws {counts['matched']} of the CT's {counts['grey']} pixels as "
                            f"the green both added over black, {out / f'layers-alone-{pane}.png'}")
    same(await press("both-last", "The CT with its layers", ctrl=True), "both-last", both, "both")

    t0 = time.monotonic()
    await page.key("4")
    await page.settle(t0, {"panes complete": ct_key, "layer": layer_key})
    await page.key("z")
    three_d = (await page.panes())["3d"]

    async def maximised(name):
        await page.settle(time.monotonic(), DRAWN)
        png, pixels = await page.picture(three_d)
        (out / f"{name}-3d.png").write_bytes(png)
        return pixels["sha256"]

    shown = await maximised("maximised")
    await page.key(" ")
    if await maximised("maximised-ct-alone") == shown:
        failures.append(f"the maximised 3D pane drew the CT alone as both, {out / 'maximised-ct-alone-3d.png'}")
    await page.key(" ")
    if await maximised("maximised-both") != shown:
        failures.append(f"the maximised 3D pane drew both again otherwise, {out / 'maximised-both-3d.png'}")
    return failures
