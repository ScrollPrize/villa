import asyncio
import base64
import json
import struct
import time
import zlib

import aiohttp

from page import CENTER_XYZ, NEUTRAL_XY, POLL_S, TIMEOUT_S, VOLUME, ct_key, lines_key, slices_key, surface_key

INK = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20231221180251/"
       "ink-detection/downsampled/PHercParis4-20231221180251-2.4um-0.22m-78keV-volume-20260411134726-20260417190342-"
       "new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg")
GRID_HW = (1598, 2103)      # the segment's x.tif; the image is its grid over the scale 0.05, at a downsampling of 8
WHEELS = 6
STEP_PX = 6
BRIGHT, DARK = 200, 40
BRIGHT_SHARE, DARK_SHARE, CORRELATION = 0.5, 0.25, 0.8
TIFF = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20231221180251/"
        "ink-detection/PHercParis4-20231221180251-2.4um-0.22m-78keV-volume-20260411134726-20260417190342-"
        "new_canon_autoresearch_recipe-tile256-stride128.tif")
SCALE = 0.05
INKY_UV = [1011.2, 755.2]   # a grid cell whose 512² full pixels about it are 48% ink, 33% background, over 4 tiles
DETAIL_PIXEL = 0.8
CATALOG_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json"
# The value the pane draws, taken out of green's blend, is at most this far from the TIFF's pixel, in rms grey levels,
# with the TIFF, where the blend's rounding keeps it near 0.6, and at most this share of how far the image alone is.
DETAIL_RMS, DETAIL_SHARE = 0.8, 0.75
OPTIONS = """(() => { const options = [...document.querySelector("#ink-catalog").options];
  return options[0].text.startsWith("Reading") || !options[0].text.includes("(") && !options[0].text.includes("none")
    ? null : options.map((option) => [option.text, option.value]) })()"""
READ = "![...document.querySelectorAll('#ink-items .layer-row-name')].some((name) => name.textContent.endsWith('…'))"
GPU = "Object.entries(window.__tecaDebug.rendererStats()).map(([p, r]) => [p, r.textures, r.geometries])"
SAMPLES = """(async ([before, after, points, location, fractions]) => {
  const rgba = async (blob) => {
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    return { width: bitmap.width, height: bitmap.height, data: context.getImageData(0, 0, bitmap.width, bitmap.height).data };
  };
  const png = async (text) => rgba(await (await fetch("data:image/png;base64," + text)).blob());
  const [b, a, image] = await Promise.all([png(before), png(after), fetch(location).then((r) => r.blob()).then(rgba)]);
  const rgb = (picture, [x, y]) => [0, 1, 2].map((c) => picture.data[4 * (y * picture.width + x) + c]);
  const grey = ([fx, fy]) => {
    const [x, y] = [fx * image.width, fy * image.height];
    let total = 0;
    for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const px = Math.min(image.width - 1, Math.max(0, Math.floor(x - 0.5) + dx));
      const py = Math.min(image.height - 1, Math.max(0, Math.floor(y - 0.5) + dy));
      total += image.data[4 * (py * image.width + px)];
    }
    return total / 4;
  };
  return points.map((point, index) => [...rgb(b, point), ...rgb(a, point), grey(fractions[index])]);
})"""
STATUS = "document.querySelector('#status').textContent"
INKED_XYZ = "20610.88, 13996.71, 44907.56"   # surface grid cell (row 759, col 1040), where the image is bright
LOW = 100
NEAR = 2
SEEN = 6
FEW = {"xy": 300, "surface": 1000, "3d": 100}
EDGE_PX = 2
DOCKED = ("xy", "surface", "3d")
ROW = "#ink-items .layer-row"
ENDS = ("Lowest pixel shown", "Highest pixel shown")
RECORD = "#performance-controls [data-record]"
EXPORT = "#performance-controls [data-export]"
MADE = "Object.entries(window.__tecaDebug.rendererStats()).map(([p, r]) => [p, r.programs, r.textures, r.geometries])"
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
  const counts = { grey: 0, cutUnder: 0, under: 0, underAsCT: 0, over: 0, overAsFull: 0, overDrawn: 0, near: 0, other: 0 };
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
      if (!same(k, c, i)) counts.overDrawn++;
    }
  }
  return counts;
})"""


async def tiff_pixels(cells_HW):
    async with aiohttp.ClientSession() as http:
        async def read(offset, length):
            async with http.get(TIFF, headers={"Range": f"bytes={offset}-{offset + length - 1}"}) as response:
                return await response.read()
        head = await read(0, 65536)
        ifd = struct.unpack_from("<I", head, 4)[0]
        tags = {}
        for entry in range(struct.unpack_from("<H", head, ifd)[0]):
            tag, kind, _, value = struct.unpack_from("<HHII", head, ifd + 2 + 12 * entry)
            tags[tag] = value & 0xFFFF if kind == 3 else value
        width, tile_h, tile_w = tags[256], tags[323], tags[322]
        columns = -(-width // tile_w)
        tiles, values = {}, []
        for row, column in cells_HW:
            index = row // tile_h * columns + column // tile_w
            if index not in tiles:
                offset, length = (struct.unpack_from("<I", head, tags[tag] + 4 * index)[0] for tag in (324, 325))
                tiles[index] = zlib.decompress(await read(offset, length))
            values.append(tiles[index][row % tile_h * tile_w + column % tile_w])
        return values


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
    if (place := await page.place("#ink-section")) != "data":
        failures.append(f"its card is on {place!r}, not the Data tab")
    await page.reveal("#ink-section")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    t0 = time.monotonic()
    await page.key("u")
    await page.settle(t0, {"surface": surface_key})
    rect = (await page.panes())["surface"]
    cx, cy = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
    await page.mouse("mouseMoved", cx, cy)
    for _ in range(WHEELS):
        await page.mouse("mouseWheel", cx, cy, deltaX=0, deltaY=-240)

    async def picture(name):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"surface": surface_key})
        png, pixels = await page.picture(rect)
        (out / f"{name}.png").write_bytes(png)
        return png, pixels["sha256"]

    async def textures(count, t0):
        while (await page.evaluate("window.__tecaDebug.rendererStats().surface.textures")) != count:
            if time.monotonic() - t0 > TIMEOUT_S:
                raise TimeoutError(f"the Surface pane did not come to {count} textures in {TIMEOUT_S} s: {await page.evaluate(GPU)}")
            await asyncio.sleep(POLL_S)

    ct_png, ct = await picture("ct")
    before = await page.evaluate(GPU)
    surface_textures = await page.evaluate("window.__tecaDebug.rendererStats().surface.textures")
    t0 = time.monotonic()
    await page.type_into("ink-location", INK)
    await page.click("#ink-add")
    await page.browser.wait_for("document.querySelector('#ink-items .layer-row') !== null", TIMEOUT_S)
    await page.choose("#ink-items .layer-row input[type=color]", "#00ff00")
    await page.choose("#ink-items .layer-row input[type=range]", "1")
    await textures(surface_textures + 1, t0)
    ink_png, ink = await picture("ink")
    added = await page.evaluate(GPU)

    base_cells = await page.config("SURFACE_BASE_VIEW_CELLS")
    view = (await page.evaluate("window.__tecaDebug.session()"))["surfaceView"]
    aspect = rect["width"] / rect["height"]
    base = base_cells / view["zoom"]
    span_uv = (base * max(1, aspect), base * max(1, 1 / aspect))
    points, fractions = [], []
    for y in range(round(rect["height"] * 0.1), round(rect["height"] * 0.9), STEP_PX):
        for x in range(round(rect["width"] * 0.1), round(rect["width"] * 0.9), STEP_PX):
            u = view["center_UV"][0] + ((x + 0.5) / rect["width"] - 0.5) * span_uv[0]
            v = view["center_UV"][1] + ((y + 0.5) / rect["height"] - 0.5) * span_uv[1]
            points.append([x, y])
            fractions.append([u / GRID_HW[1], v / GRID_HW[0]])
    arguments = [base64.b64encode(ct_png).decode(), base64.b64encode(ink_png).decode(), points, INK, fractions]
    samples = await page.evaluate(f"{SAMPLES}({json.dumps(arguments)})")
    shares, values = [], []
    for r0, g0, b0, r1, g1, b1, value in samples:
        if r0 + g0 + b0 == 0:
            continue
        grey = (r0 + g0 + b0) / 3
        shares.append((abs(r1 - r0) + abs(g1 - g0) + abs(b1 - b0)) / (255 + grey))
        values.append(value)
    bright = [share for share, value in zip(shares, values) if value >= BRIGHT]
    dark = [share for share, value in zip(shares, values) if value <= DARK]
    mean = lambda items: sum(items) / len(items)
    if len(bright) < 100 or len(dark) < 100:
        failures.append(f"the pane shows {len(bright)} samples of ink and {len(dark)} of background, not 100 of each")
    else:
        ms, mv = mean(shares), mean(values)
        covariance = sum((s - ms) * (v - mv) for s, v in zip(shares, values))
        spread = (sum((s - ms) ** 2 for s in shares) * sum((v - mv) ** 2 for v in values)) ** 0.5
        correlation = covariance / spread
        print(f"green's share over ink {mean(bright):.3f} ({len(bright)}), over background {mean(dark):.3f} ({len(dark)}), "
              f"correlation with the image {correlation:.3f} over {len(shares)} samples")
        if mean(bright) < BRIGHT_SHARE or mean(dark) > DARK_SHARE or correlation < CORRELATION:
            failures.append(f"green's share is {mean(bright):.3f} over ink and {mean(dark):.3f} over background, "
                            f"correlation {correlation:.3f}: not at least {BRIGHT_SHARE}, at most {DARK_SHARE} and "
                            f"{CORRELATION}, {out / 'ink.png'}")

    async def press(name, said, ctrl=False):
        await page.key(" ", ctrl=ctrl)
        if not (status := await page.evaluate(STATUS)).startswith(said):
            failures.append(f"after {name} the status bar says {status!r}, not {said!r}")
        return (await picture(name))[1]

    await page.evaluate("document.activeElement.blur(), true")
    if await press("ct-alone", "The CT alone") != ct:
        failures.append(f"Space did not draw the CT alone, {out / 'ct-alone.png'}")
    if await press("both-again", "The CT with its layers") != ink:
        failures.append(f"Space again did not draw the image over the CT, {out / 'both-again.png'}")
    if (alone := await press("image-alone", "The layers and images alone", ctrl=True)) in (ct, ink):
        failures.append(f"Ctrl+Space drew the pane as {'the CT alone' if alone == ct else 'both'}, "
                        f"{out / 'image-alone.png'}")
    if await press("both-last", "The CT with its layers", ctrl=True) != ink:
        failures.append(f"Ctrl+Space again did not draw the image over the CT, {out / 'both-last.png'}")

    await page.choose("#ink-items .layer-row input[type=checkbox]", checked=False)
    _, hidden = await picture("hidden")
    if hidden != ct:
        failures.append(f"hiding the image did not give the pane back, {out / 'hidden.png'}")
    await page.choose("#ink-items .layer-row input[type=checkbox]", checked=True)
    _, shown = await picture("shown")
    if shown != ink:
        failures.append(f"showing the image again drew another picture, {out / 'shown.png'}")

    t0 = time.monotonic()
    await page.open(await page.address())
    await page.settle(t0, {"surface": surface_key})
    await page.browser.wait_for(READ, TIMEOUT_S)
    _, reloaded = await picture("reloaded")
    [entry] = (await page.evaluate("window.__tecaDebug.session()"))["extensions"]["ink"]
    if reloaded != ink:
        failures.append(f"the reloaded page drew another picture, {out / 'reloaded.png'}")

    # A reload draws with other GPU objects than the page before it, so each page's difference is compared.
    drawn = await page.evaluate(GPU)
    await page.click("#ink-items .layer-row .icon-btn")
    _, removed = await picture("removed")
    if removed != ct:
        failures.append(f"removing the image did not give the pane back, {out / 'removed.png'}")
    freed = await page.evaluate(GPU)
    if [[pane, textures + 1, geometries] for pane, textures, geometries in freed] != drawn:
        failures.append(f"removing the image left textures and geometries {freed}, not one texture fewer than {drawn}")
    if [[pane, textures + 1, geometries] for pane, textures, geometries in before] != added:
        failures.append(f"adding the image took textures and geometries {added}, not one texture more than {before}")

    session = await page.evaluate("window.__tecaDebug.session()")
    zoom = base_cells / SCALE / (DETAIL_PIXEL * min(rect["width"], rect["height"]))
    session["surfaceView"].update(center_UV=INKY_UV, zoom=zoom)
    image = {"location": INK, "tiff": None, "surface": entry["surface"], "color": "#00ff00", "opacity": 1,
             "visible": True, "volume": False, "steps": 4, "range": [0, 255]}

    async def opened(name, images):
        session["extensions"]["ink"] = images
        t0 = time.monotonic()
        await page.open_session(session)
        await page.settle(t0, {"surface": surface_key})
        await textures(close_textures + len(images) + sum(i["tiff"] is not None for i in images), t0)
        print(f"{name}: drawn {time.monotonic() - t0:.1f} s after the page opened")
        return await picture(name)

    session["extensions"]["ink"] = []
    t0 = time.monotonic()
    await page.open_session(session)
    await page.settle(t0, {"surface": surface_key})
    close_textures = await page.evaluate("window.__tecaDebug.rendererStats().surface.textures")
    close_png, close = await picture("close")
    alone_png, _ = await opened("close-image", [image])
    detail_png, _ = await opened("close-tiff", [{**image, "tiff": TIFF}])
    read = [request for request in page.browser.requests.values() if request["url"] == TIFF]
    print(f"the TIFF: {len(read)} range reads, {sum(r['bytes'] for r in read)} bytes")

    span = base_cells / zoom
    span_uv = (span * max(1, aspect), span * max(1, 1 / aspect))
    points, cells = [], []
    for y in range(round(rect["height"] * 0.1), round(rect["height"] * 0.9), STEP_PX):
        for x in range(round(rect["width"] * 0.1), round(rect["width"] * 0.9), STEP_PX):
            u = INKY_UV[0] + ((x + 0.5) / rect["width"] - 0.5) * span_uv[0]
            v = INKY_UV[1] + ((y + 0.5) / rect["height"] - 0.5) * span_uv[1]
            points.append([x, y])
            cells.append([int(v / SCALE), int(u / SCALE)])
    values = await tiff_pixels(cells)
    rms = {}
    for name, png in [("image", alone_png), ("tiff", detail_png)]:
        pictures = [base64.b64encode(close_png).decode(), base64.b64encode(png).decode()]
        samples = await page.evaluate(f"{SAMPLES}({json.dumps([*pictures, points, INK, [[0, 0]] * len(points)])})")
        # Green at full opacity over the grey g0 draws g0 + (255 - g0) * value / 255; over a grey past 200 the
        # division would grow the blend's rounding past a grey level.
        errors = [255 * (g1 - g0) / (255 - g0) - value
                  for (r0, g0, b0, _, g1, _, _), value in zip(samples, values) if 0 < r0 + g0 + b0 and g0 <= 200]
        rms[name] = (sum(error * error for error in errors) / len(errors)) ** 0.5
        print(f"with the {name}: the drawn value is {rms[name]:.3f} rms grey levels from the TIFF's, "
              f"over {len(errors)} samples")
    if rms["tiff"] > DETAIL_RMS or rms["tiff"] > DETAIL_SHARE * rms["image"]:
        failures.append(f"with its TIFF the image is drawn {rms['tiff']:.3f} rms grey levels from the TIFF's pixels, "
                        f"and {rms['image']:.3f} without it: not at most {DETAIL_RMS} and {DETAIL_SHARE} of it, "
                        f"{out / 'close-tiff.png'}")
    await page.click("#ink-items .layer-row .icon-btn")
    _, removed = await picture("close-removed")
    if removed != close:
        failures.append(f"removing the image and its TIFF did not give the pane back, {out / 'close-removed.png'}")
    left = await page.evaluate("window.__tecaDebug.rendererStats().surface.textures")
    if left != close_textures:
        failures.append(f"removing the image and its TIFF left {left} textures, not {close_textures}")

    if reads := [r for r in page.browser.requests.values() if r["url"] == CATALOG_URL]:
        failures.append(f"the catalog was read before the choice was pressed: {reads}")
    await page.reveal("#ink-section")
    t0 = time.monotonic()
    await page.click("#ink-catalog")
    options = await page.wait_text(OPTIONS, lambda value: value is not None, TIMEOUT_S)
    print(f"the choice listed {len(options) - 1} {time.monotonic() - t0:.2f} s after its press: {options[0][0]!r}")
    await page.key("Escape")
    if (offered := [value for _, value in options[1:]]) != [INK]:
        failures.append(f"the choice offers {offered}, not {[INK]}")
    else:
        await page.choose("#ink-catalog", INK)
        filled = await page.evaluate("['#ink-surface', '#ink-location', '#ink-tiff'].map((id) => "
                                     "document.querySelector(id).value)")
        if filled != [entry["surface"], INK, TIFF]:
            failures.append(f"picking {INK} filled the surface and the fields with {filled}, not {entry['surface']}, "
                            f"it and {TIFF}")
        t0 = time.monotonic()
        await page.click("#ink-add")
        await textures(close_textures + 2, t0)
        _, picked = await picture("close-picked")
        if picked == close:
            failures.append(f"the image picked from the choice left the pane as it was, {out / 'close-picked.png'}")
        if len(reads := [r for r in page.browser.requests.values() if r["url"] == CATALOG_URL]) != 1:
            failures.append(f"the catalog was read {len(reads)} times, not once: {reads}")

    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.type_into("position-number", INKED_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", "1")
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

    async def pictures(name):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"panes complete": ct_key, "surface": surface_key, "layers": layers_key,
                                             "drawn": drawn_key})
        rects = await page.panes()
        shot = {}
        for pane in DOCKED:
            r = rects[pane]
            rect = {"x": r["x"] + EDGE_PX, "y": r["y"] + EDGE_PX, "width": r["width"] - 2 * EDGE_PX,
                    "height": r["height"] - 2 * EDGE_PX}
            png, pixels = await page.picture(rect)
            (out / f"{name}-{pane}.png").write_bytes(png)
            shot[pane] = (png, pixels["sha256"])
        return shot

    async def report():
        await page.click(RECORD)
        path = await page.download(EXPORT, "viewer-performance.json")
        recorded = json.loads(path.read_text())
        path.unlink()
        return recorded

    inked = await pictures("inked")
    await page.reveal("#ink-section")
    await page.type_into("ink-location", INK)
    await page.click("#ink-add")
    await page.browser.wait_for(f"document.querySelector('{ROW}') !== null", TIMEOUT_S)
    await page.choose(f"{ROW} input[type=color]", "#00ff00")
    await page.choose(f"{ROW} input[type=range]", "1")
    await page.browser.wait_for(READ, TIMEOUT_S)
    await page.choose("#ink-items input[aria-label='Show ink image in the volume and the 3D pane']", checked=True)
    full = await pictures("full")
    await page.reveal(RECORD)
    await page.click(RECORD)
    gpu = await page.evaluate(MADE)
    changed = time.monotonic()
    await page.choose(f"#ink-items input[aria-label='{ENDS[0]}']", str(LOW))
    cut = await pictures("cut")
    recorded = await report()
    if reads := [r["url"] for r in page.browser.requests.values() if r["start"] >= changed]:
        failures.append(f"the change of range read {len(reads)} files, as {reads[0]}")
    if (uploads := recorded["metrics"]["uploads"])["bytes"] != 0:
        failures.append(f"the change of range uploaded {uploads['bytes']} bytes in {uploads['jobs']} uploads")
    if (after := await page.evaluate(MADE)) != gpu:
        failures.append(f"the change of range changed the panes' GPU programs, textures and geometries from {gpu} to "
                        f"{after}")
    for pane in DOCKED:
        arguments = [base64.b64encode(shot[pane][0]).decode() for shot in (inked, full, cut)] + [LOW, NEAR, SEEN]
        counts = await page.evaluate(f"{CUT}({json.dumps(arguments)})")
        print(f"the cut on {pane}: {counts}")
        # The 3D pane draws the image on the surface's mesh, whose cells lie over each other along a ray edge on: their
        # values under LOW add up over it, so the cut need draw only some of the pixels over LOW there.
        if counts["under"] < FEW[pane] or counts["over"] < FEW[pane]:
            failures.append(f"on {pane} the full range draws {counts['under']} pixels under {LOW} and {counts['over']} "
                            f"over it, not {FEW[pane]} of each, {out / f'full-{pane}.png'}")
        elif counts["cutUnder"] != 0:
            failures.append(f"on {pane} the cut draws {counts['cutUnder']} pixels by a value under {LOW}, "
                            f"{out / f'cut-{pane}.png'}")
        elif counts["underAsCT"] != counts["under"]:
            failures.append(f"on {pane} the cut draws {counts['underAsCT']} of the {counts['under']} pixels under {LOW} "
                            f"as the CT, {out / f'cut-{pane}.png'}")
        elif counts["overAsFull"] != counts["over"] and (pane != "3d" or counts["overDrawn"] < FEW[pane]):
            failures.append(f"on {pane} the cut draws {counts['overAsFull']} of the {counts['over']} pixels over {LOW} "
                            f"as the full range did and {counts['overDrawn']} otherwise than the CT, "
                            f"{out / f'cut-{pane}.png'}")
    t0 = time.monotonic()
    await page.open(await page.address())
    await page.settle(t0, {"panes complete": ct_key, "surface": surface_key})
    await page.browser.wait_for(f"document.querySelector('{ROW}') !== null && {READ}", TIMEOUT_S)
    ends = await page.evaluate(f"""[...document.querySelectorAll("#ink-items input[type=number]")]
      .filter((input) => {json.dumps(ENDS)}.includes(input.getAttribute("aria-label"))).map((input) => input.value)""")
    if ends != [str(LOW), "255"]:
        failures.append(f"the reloaded row's range is {ends}, not [{LOW}, 255]")
    reloaded = await pictures("reloaded")
    for pane in DOCKED:
        if reloaded[pane][1] != cut[pane][1]:
            failures.append(f"the reloaded page drew {pane} otherwise than the cut, {out / f'reloaded-{pane}.png'}")
    return failures
