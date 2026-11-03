import base64
import json
import time

from page import CENTER_XYZ, NEUTRAL_XY, TIMEOUT_S, VOLUME, ct_key, lines_key, slices_key, surface_key

BUCKET_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
CATALOG_URL = f"{BUCKET_URL}/metadata.json"
PREDICTIONS = f"{BUCKET_URL}/PHercParis4/representations/predictions"
INK = f"{PREDICTIONS}/ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/"
COARSE = f"{PREDICTIONS}/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/"
ZOOM = "1"
GREEN = "#00ff00"
PANES = ("xy", "surface", "3d")
LOW = 100
NEAR = 2
SEEN = 6
FEW = {"xy": 300, "surface": 1000, "3d": 100}
EDGE_PX = 2
ROW = "#layers-items .layer-row"
ENDS = ("Lowest value shown", "Highest value shown")
RECORD = "#performance-controls [data-record]"
EXPORT = "#performance-controls [data-export]"
OPTIONS = """(() => { const options = [...document.querySelector("#layers-catalog").options];
  return options[0].text.startsWith("Reading") || !options[0].text.includes("(") && !options[0].text.includes("none")
    ? null : options.map((option) => [option.text, option.value]) })()"""
GPU = "Object.entries(window.__tecaDebug.rendererStats()).map(([p, r]) => [p, r.programs, r.textures, r.geometries])"
CUT = """(async ([ct, full, cut, low, near, seen]) => {
  const rgba = async (text) => {
    const blob = await (await fetch("data:image/png;base64," + text)).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    return context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  };
  const [c, f, k] = await Promise.all([ct, full, cut].map(rgba));
  const same = (p, q, i) => p[i] === q[i] && p[i + 1] === q[i + 1] && p[i + 2] === q[i + 2];
  const counts = { grey: 0, cutUnder: 0, under: 0, underAsCT: 0, over: 0, overAsFull: 0, near: 0, other: 0 };
  for (let i = 0; i < c.length; i += 4) {
    if (c[i] !== c[i + 1] || c[i] !== c[i + 2]) continue;
    counts.grey++;
    // Green at full opacity over the grey g by the value v draws red and blue g(1 - v/255) and green that and v.
    const value = f[i + 1] - f[i];
    if (!same(k, c, i) && k[i + 1] - k[i] < low - near) counts.cutUnder++;
    if (Math.abs(f[i] - f[i + 2]) > 1) counts.other++;
    else if (Math.abs(value - low) < near) counts.near++;
    else if (value < low) {
      if (value <= seen) continue;
      counts.under++;
      if (same(k, c, i)) counts.underAsCT++;
    } else {
      counts.over++;
      if (same(k, f, i)) counts.overAsFull++;
    }
  }
  return counts;
})"""


def layers_key(s):
    keys = None if s is None else [slices_key(layer) for layer in s["layers"]]
    return None if keys is None or None in keys else json.dumps(keys)


def drawn_key(s):
    return None if s is None else s["frames"]


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    if (place := await page.place("#layers-section")) != "data":
        failures.append(f"its card is on {place!r}, not the Data tab")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", ZOOM)
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    t0 = time.monotonic()
    await page.key("u")
    await page.settle(t0, {"surface": surface_key})
    for selector, value, checked in (("#surface-layout-mode", "below", None), ("[data-surface-plane='3d']", None, True)):
        t0 = time.monotonic()
        await page.choose(selector, value, checked)
        await page.browser.wait_for("!document.querySelector('#surface-layout-mode').disabled", TIMEOUT_S)
        await page.settle(t0, {"panes complete": ct_key, "surface": surface_key})
    await page.reveal("#layers-section")

    async def pictures(name):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"panes complete": ct_key, "surface": surface_key, "layers": layers_key,
                                             "drawn": drawn_key})
        rects = await page.panes()
        shot = {}
        for pane in PANES:
            r = rects[pane]
            rect = {"x": r["x"] + EDGE_PX, "y": r["y"] + EDGE_PX, "width": r["width"] - 2 * EDGE_PX,
                    "height": r["height"] - 2 * EDGE_PX}
            png, pixels = await page.picture(rect)
            (out / f"{name}-{pane}.png").write_bytes(png)
            shot[pane] = (png, pixels["sha256"])
        return shot

    def same(shot, name, reference, drew):
        for pane in PANES:
            if shot[pane][1] != reference[pane][1]:
                failures.append(f"{name} did not draw {pane} as {drew}, {out / f'{name}-{pane}.png'}")

    async def report():
        await page.click(RECORD)
        path = await page.download(EXPORT, "viewer-performance.json")
        recorded = json.loads(path.read_text())
        path.unlink()
        return recorded

    ct = await pictures("ct")
    if reads := [r for r in page.browser.requests.values() if r["url"] == CATALOG_URL]:
        failures.append(f"the catalog was read before the choice was pressed: {reads}")
    t0 = time.monotonic()
    await page.click("#layers-catalog")
    options = await page.wait_text(OPTIONS, lambda value: value is not None, TIMEOUT_S)
    print(f"the choice listed {len(options) - 1} {time.monotonic() - t0:.2f} s after its press: {options[0][0]!r}")
    await page.key("Escape")
    if INK not in (offered := [value for _, value in options[1:]]) or COARSE in offered:
        failures.append(f"the choice offers {offered}: not {INK}, or also {COARSE}")
        return failures
    await page.choose("#layers-catalog", INK)
    if (filled := await page.evaluate("document.querySelector('#layers-location').value")) != INK:
        failures.append(f"picking {INK} filled the layer's field with {filled!r}")
    before = await page.evaluate(GPU)
    await page.click("#layers-add")
    ink = await pictures("ink")
    if len(reads := [r for r in page.browser.requests.values() if r["url"] == CATALOG_URL]) != 1:
        failures.append(f"the catalog was read {len(reads)} times, not once: {reads}")
    added = await page.evaluate(GPU)
    for pane in PANES:
        if ink[pane][1] == ct[pane][1]:
            failures.append(f"the ink layer left {pane} as it was, {out / f'ink-{pane}.png'}")

    await page.choose(f"{ROW} input[type=color]", GREEN)
    recoloured = await pictures("recoloured")
    for pane in PANES:
        if recoloured[pane][1] in (ct[pane][1], ink[pane][1]):
            drew = "CT" if recoloured[pane][1] == ct[pane][1] else "first colour"
            failures.append(f"colour {GREEN} drew the {drew} on {pane}, {out / f'recoloured-{pane}.png'}")
    await page.choose(f"{ROW} input[type=range]", "1")
    full = await pictures("full")

    await page.reveal(RECORD)
    await page.click(RECORD)
    gpu = await page.evaluate(GPU)
    changed = time.monotonic()
    await page.choose(f"#layers-items input[aria-label='{ENDS[0]}']", str(LOW))
    cut = await pictures("cut")
    recorded = await report()
    if reads := [r["url"] for r in page.browser.requests.values() if r["start"] >= changed]:
        failures.append(f"the change of range read {len(reads)} files, as {reads[0]}")
    if (uploads := recorded["metrics"]["uploads"])["bytes"] != 0:
        failures.append(f"the change of range uploaded {uploads['bytes']} bytes in {uploads['jobs']} uploads")
    if (after := await page.evaluate(GPU)) != gpu:
        failures.append(f"the change of range changed the panes' GPU programs, textures and geometries from {gpu} to "
                        f"{after}")
    for pane in PANES:
        arguments = [base64.b64encode(shot[pane][0]).decode() for shot in (ct, full, cut)] + [LOW, NEAR, SEEN]
        counts = await page.evaluate(f"{CUT}({json.dumps(arguments)})")
        print(f"the cut on {pane}: {counts}")
        if counts["under"] < FEW[pane] or counts["over"] < FEW[pane]:
            failures.append(f"on {pane} the full range draws {counts['under']} pixels under {LOW} and {counts['over']} "
                            f"over it, not {FEW[pane]} of each, {out / f'full-{pane}.png'}")
        elif counts["cutUnder"] != 0:
            failures.append(f"on {pane} the cut draws {counts['cutUnder']} pixels by a value under {LOW}, "
                            f"{out / f'cut-{pane}.png'}")
        elif counts["underAsCT"] != counts["under"]:
            failures.append(f"on {pane} the cut draws {counts['underAsCT']} of the {counts['under']} pixels under {LOW} "
                            f"as the CT, {out / f'cut-{pane}.png'}")
        elif counts["overAsFull"] != counts["over"]:
            failures.append(f"on {pane} the cut draws {counts['overAsFull']} of the {counts['over']} pixels over {LOW} "
                            f"as the full range did, {out / f'cut-{pane}.png'}")

    t0 = time.monotonic()
    await page.open(await page.address())
    await page.settle(t0, {"panes complete": ct_key, "surface": surface_key})
    await page.browser.wait_for(f"document.querySelector('{ROW}') !== null", TIMEOUT_S)
    ends = await page.evaluate(f"""[...document.querySelectorAll("#layers-items input[type=number]")]
      .filter((input) => {json.dumps(ENDS)}.includes(input.getAttribute("aria-label"))).map((input) => input.value)""")
    if ends != [str(LOW), "255"]:
        failures.append(f"the reloaded row's range is {ends}, not [{LOW}, 255]")
    same(await pictures("reloaded"), "the reloaded page", cut, "the cut")
    held = await page.evaluate(GPU)

    await page.reveal("#layers-section")
    await page.choose(f"{ROW} input[type=checkbox]", checked=False)
    same(await pictures("hidden"), "hiding the layer", ct, "the CT")
    await page.choose(f"{ROW} input[type=checkbox]", checked=True)
    same(await pictures("shown"), "showing the layer again", cut, "the cut")
    await page.click(f"{ROW} button")
    same(await pictures("removed"), "removing the layer", ct, "the CT")
    took = [[a[2] - b[2], a[3] - b[3]] for a, b in zip(added, before)]
    freed = [[h[2] - r[2], h[3] - r[3]] for h, r in zip(held, await page.evaluate(GPU))]
    if freed != took:
        failures.append(f"removing the layer freed {freed} textures and geometries per pane, not the {took} adding it took")
    return failures
