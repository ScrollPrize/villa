import asyncio
import base64
import gzip
import json
import time
import zipfile

from page import CENTER_XYZ, TIMEOUT_S, VOLUME, ct_key, lines_key, slices_key, surface_key

ZOOM = "1"
RADIUS = "32"
LARGE_RADIUS = "256"  # a stroke whose chunks the browser takes about 0.4 s to pack
RELOAD_MS = 8
PACE_S = 0.016
REST_S = 0.05
PACKED_S = 1.0        # the chunks follow an edit by 0.56 s, and an entry asked for as they are written waits for them
STROKE_PX = 120
PAN_PX = 120
STEP_PX = 10
GATED_PX = 100
WHEELS = 20
OFF_FOCUS_PX = 30
MARGIN = 24
GATED_SHARE = 0.9
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-labels.zarr.zip"
OTHER = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/"
         "ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/")
SECOND_S = 30
BACKLOG_WHEELS = 16  # notches the Surface is zoomed out for the backlog, whose strokes take the labels to 2,288 chunks
BACKLOG_PX = (-GATED_PX, 0, GATED_PX)
KEPT = ", kept in this browser."
READY = "!document.querySelector('#paint-tool').disabled"
NOTE = "document.querySelector('#paint-note').textContent"
STATUS = "document.querySelector('#status').textContent"
SECOND_DOCUMENT = "document.querySelector('#second')?.contentDocument"
SECOND_NOTE = f"({SECOND_DOCUMENT}?.querySelector('#paint-note')?.textContent ?? '')"
HANDED = ", 3 to undo"
BACKLOGGED = f", {2 + len(BACKLOG_PX)} to undo."
RECORD_NOTES = ("(() => { const note = document.querySelector('#second').contentDocument.querySelector('#paint-note'); "
                "window.__secondNotes = []; new MutationObserver(() => window.__secondNotes.push(note.textContent))"
                ".observe(note, { childList: true, characterData: true, subtree: true }); return true })()")
PIXEL = """(async ([png, x, y]) => {
  const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
    { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  return [...context.getImageData(x, y, 1, 1).data.slice(0, 3)];
})"""
COMPARE = """(async (pngs) => {
  const rgba = async (png) => {
    const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
      { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    return context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  };
  const [a, b, c] = await Promise.all(pngs.map(rgba));
  const differ = (p, q, i) => p[i] !== q[i] || p[i + 1] !== q[i + 1] || p[i + 2] !== q[i + 2];
  const greys = Array.from({ length: 256 }, () => [0, 0]);
  let region = 0, inside = 0, outside = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (!differ(a, b, i)) {
      if (differ(b, c, i)) outside++;
      continue;
    }
    region++;
    if (differ(b, c, i)) inside++;
    greys[a[i]][0]++;
    if (differ(a, c, i)) greys[a[i]][1]++;
  }
  return { region, inside, outside, greys };
})"""


def layer_key(s):
    return None if s is None or len(s["layers"]) != 1 else slices_key(s["layers"][0])


def no_layer_key(s):
    return None if s is None or s["layers"] else "none"


def reopened(notes):
    if "" not in notes:
        return None
    return next((text for text in notes[notes.index("") + 1:] if text != ""), None)


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    if (place := await page.place("#paint-section")) != "Paint":
        failures.append(f"its card is on {place!r}, not a Paint tab")
    await page.reveal("#paint-section")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", ZOOM)
    await page.settle(t0, {"panes complete": ct_key})
    rect = (await page.panes())["xy"]
    x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
    painted_keys = {"panes complete": ct_key, "labels": layer_key}

    async def picture(name, keys, area=rect, off_x=0):
        await page.mouse("mouseMoved", 150, 950)
        await page.settle(time.monotonic(), keys)
        png, pixels = await page.picture(area)
        (out / f"{name}.png").write_bytes(png)
        at = [base64.b64encode(png).decode(), round(area["width"] / 2) + off_x, round(area["height"] / 2)]
        centre = await page.evaluate(f"{PIXEL}({json.dumps(at)})")
        return png, pixels["sha256"], centre

    async def compare(a, b, c):
        return await page.evaluate(f"{COMPARE}({json.dumps([base64.b64encode(png).decode() for png in (a, b, c)])})")

    async def kept():
        await page.wait_text(NOTE, lambda note: note.endswith(KEPT), 60)

    async def begun(edit):
        if (await page.evaluate(NOTE)).endswith(KEPT):
            failures.append(f"the note says kept as a {edit} begins, before the journal holds it")

    async def stroke(at_x, at_y, then=None, paced=False):
        await page.mouse("mouseMoved", at_x - STROKE_PX // 2, at_y)
        await page.mouse("mousePressed", at_x - STROKE_PX // 2, at_y, button="left", buttons=1, clickCount=1)
        await begun("stroke")
        for step_x in range(at_x - STROKE_PX // 2 + STEP_PX, at_x + STROKE_PX // 2 + 1, STEP_PX):
            await page.mouse("mouseMoved", step_x, at_y, button="left", buttons=1)
            if paced:
                await asyncio.sleep(PACE_S)
        if paced:
            await asyncio.sleep(REST_S)
        await page.mouse("mouseReleased", at_x + STROKE_PX // 2, at_y, button="left", buttons=0, clickCount=1)
        return await (kept() if then is None else then())

    async def reload(query):
        await page.open(query)
        await page.browser.wait_for(READY, TIMEOUT_S)

    async def soon(edit, query):
        await asyncio.sleep(RELOAD_MS / 1000)
        await reload(query)
        note = await page.evaluate(NOTE)
        print(f"reloaded {RELOAD_MS} ms after the {edit}: {note!r}")
        if "made after them" not in note:
            failures.append(f"the reload {RELOAD_MS} ms after the {edit} brought back no edit past the chunks, "
                            f"{note!r}")

    async def turn(selector):
        await page.click(selector)
        await page.browser.wait_for(READY, TIMEOUT_S)

    async def second(start):
        deadline = time.monotonic() + SECOND_S
        while not (note := await page.evaluate(SECOND_NOTE)).startswith(start) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return note

    ct_png, ct, ct_centre = await picture("ct", {"panes complete": ct_key})
    await page.choose("#paint-radius", RADIUS)
    await page.click("#paint-tool")
    await stroke(x, y)
    painted_png, painted, centre = await picture("painted", painted_keys)
    if painted == ct:
        failures.append(f"the stroke left xy as it was, {out / 'painted.png'}")
    # Label 1 is [0.3, 0.504, 1] at 0.6 over grey: blue exceeds red by 0.6 × 0.7 × 255, whatever the grey.
    if centre[2] - centre[0] < 80 or ct_centre[2] != ct_centre[0]:
        failures.append(f"the focus is {centre} painted and {ct_centre} before, not label 1's blue over grey")

    focus = "window.__tecaDebug.session().center_ZYX"
    start_ZYX = await page.evaluate(focus)
    await page.key(" ", down_only=True)
    await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
    await page.mouse("mouseMoved", x + PAN_PX, y, button="left", buttons=1)
    panned_ZYX = await page.evaluate(focus)
    await page.mouse("mouseMoved", x, y, button="left", buttons=1)
    await page.mouse("mouseReleased", x, y, button="left", buttons=0, clickCount=1)
    await page.browser.send("Input.dispatchKeyEvent",
                            {"type": "keyUp", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32})
    if panned_ZYX == start_ZYX:
        failures.append(f"a drag on xy with Space held left the focus at {start_ZYX}: it did not pan")
    if (await picture("held", painted_keys))[1] != painted:
        failures.append(f"a drag with Space held did not leave xy as the stroke drew it, {out / 'held.png'}")
    await page.key(" ")
    if not (status := await page.evaluate(STATUS)).startswith("The CT alone"):
        failures.append(f"after a tap of Space with the brush set the status bar says {status!r}")
    if (await picture("tapped", {"panes complete": ct_key}))[1] != ct:
        failures.append(f"a tap of Space with the brush set did not draw xy as the CT alone, {out / 'tapped.png'}")
    await page.key(" ")

    await turn("#paint-undo")
    if (await picture("undone", painted_keys))[1] != ct:
        failures.append(f"the undo did not give xy back, {out / 'undone.png'}")
    await turn("#paint-redo")
    if (await picture("redone", painted_keys))[1] != painted:
        failures.append(f"the redo drew another picture than the stroke, {out / 'redone.png'}")
    await turn("#paint-undo")
    await stroke(x, y)
    if (await picture("again", painted_keys))[1] != painted:
        failures.append(f"the same stroke painted again drew another picture, {out / 'again.png'}")

    await page.click("#paint-class-2")
    await page.choose("#paint-fill", checked=True)
    await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
    await begun("fill")
    await page.mouse("mouseReleased", x, y, button="left", buttons=0, clickCount=1)
    await kept()
    filled_png, filled, _ = await picture("filled", painted_keys)
    fill = await compare(ct_png, painted_png, filled_png)
    print(f"the fill changed {fill['inside']} of the band's {fill['region']} pixels and {fill['outside']} off it")
    if fill["inside"] != fill["region"] or fill["outside"] != 0:
        failures.append(f"the fill changed {fill['inside']} of the band's {fill['region']} pixels and "
                        f"{fill['outside']} off it, not all and none, {out / 'filled.png'}")
    await page.choose("#paint-fill", checked=False)
    await turn("#paint-undo")
    if (await picture("fill-undone", painted_keys))[1] != painted:
        failures.append(f"the fill's undo did not give the stroke back, {out / 'fill-undone.png'}")

    await stroke(x, y + GATED_PX)
    wide_png, _, _ = await picture("ungated", painted_keys)
    await turn("#paint-undo")
    greys = (await compare(painted_png, wide_png, wide_png))["greys"]
    total = sum(count for count, _ in greys)
    threshold = next(grey for grey in range(256) if sum(count for count, _ in greys[:grey + 1]) * 2 >= total)
    await page.choose("#paint-gate-value", str(threshold))
    await page.choose("#paint-gate", checked=True)
    await stroke(x, y + GATED_PX)
    gated_png, _, _ = await picture("gated", painted_keys)
    greys = (await compare(painted_png, wide_png, gated_png))["greys"]
    bright = [greys[grey] for grey in range(min(256, threshold + MARGIN), 256)]
    dark = [greys[grey] for grey in range(max(0, threshold - MARGIN + 1))]
    share = lambda bins: sum(changed for _, changed in bins) / max(1, sum(count for count, _ in bins))
    print(f"the gate at {threshold} labelled {share(bright):.3f} of {sum(c for c, _ in bright)} brighter pixels and "
          f"{share(dark):.3f} of {sum(c for c, _ in dark)} darker")
    if share(bright) < GATED_SHARE or share(dark) > 1 - GATED_SHARE:
        failures.append(f"the gate at {threshold} labelled {share(bright):.3f} of the pixels brighter by {MARGIN} and "
                        f"{share(dark):.3f} of those darker, not at least {GATED_SHARE} and at most "
                        f"{1 - GATED_SHARE:.1f}, {out / 'gated.png'}")
    await page.choose("#paint-gate", checked=False)
    await turn("#paint-undo")
    await page.click("#paint-class-1")
    if (await picture("gate-undone", painted_keys))[1] != painted:
        failures.append(f"the gated stroke's undo did not give the stroke back, {out / 'gate-undone.png'}")

    await page.choose("#paint-radius", LARGE_RADIUS)
    await stroke(x, y)
    _, large, _ = await picture("large", painted_keys)
    await asyncio.sleep(PACKED_S)
    query = await page.address()
    await page.click("#paint-undo")
    await soon("large stroke's undo", query)
    if (await picture("undo-reloaded", painted_keys))[1] != painted:
        failures.append(f"the reload {RELOAD_MS} ms after the large stroke's undo brought back another picture, "
                        f"{out / 'undo-reloaded.png'}")
    await page.click("#paint-tool")
    await page.choose("#paint-radius", LARGE_RADIUS)
    await asyncio.sleep(PACKED_S)
    query = await page.address()
    await stroke(x, y, lambda: soon("large stroke's release", query), paced=True)
    if (await picture("stroke-reloaded", painted_keys))[1] != large:
        failures.append(f"the reload {RELOAD_MS} ms after the large stroke brought back another picture, "
                        f"{out / 'stroke-reloaded.png'}")

    zip_path = await page.download("#paint-save", SAVED)
    (out / SAVED).write_bytes(zip_path.read_bytes())
    ct_zattrs, ct_zarray = await page.evaluate(f"Promise.all(['.zattrs', '0/.zarray'].map((name) => "
                                               f"fetch({json.dumps(VOLUME)} + name).then((response) => response.json())))")
    with zipfile.ZipFile(zip_path) as saved:
        zattrs, zarray = json.loads(saved.read(".zattrs")), json.loads(saved.read("0/.zarray"))
        scales = [[t["scale"] for t in d["coordinateTransformations"] if t["type"] == "scale"]
                  for d in zattrs["multiscales"][0]["datasets"]]
        ct_scales = [[t["scale"] for t in d["coordinateTransformations"] if t["type"] == "scale"]
                     for d in ct_zattrs["multiscales"][0]["datasets"]]
        if "image-label" not in zattrs or scales != ct_scales or zarray["shape"] != ct_zarray["shape"]:
            failures.append(f"the zip's levels {scales} of {zarray['shape']} are not the CT's {ct_scales} of {ct_zarray['shape']}")
        chunks_ZYX = zarray["chunks"]
        voxel_ZYX = [round(float(value)) for value in reversed(CENTER_XYZ.split(","))]
        name = "0/" + "/".join(str(v // c) for v, c in zip(voxel_ZYX, chunks_ZYX))
        at = ((voxel_ZYX[0] % chunks_ZYX[0]) * chunks_ZYX[1] + voxel_ZYX[1] % chunks_ZYX[1]) * chunks_ZYX[2] + voxel_ZYX[2] % chunks_ZYX[2]
        label = gzip.decompress(saved.read(name))[at] if name in saved.namelist() else None
        if label != 1:
            failures.append(f"the zip holds {label} at the focus voxel {voxel_ZYX}, in {name}, not 1")

    await page.click("#paint-clear")
    _, cleared, _ = await picture("cleared", {"panes complete": ct_key, "no labels": no_layer_key})
    if len(page.browser.dialogs) != 1:
        failures.append(f"Clear all asked {page.browser.dialogs}, not once")
    if cleared != ct:
        failures.append(f"clearing did not give xy back, {out / 'cleared.png'}")
    await reload(await page.address())
    await picture("cleared-reloaded", {"panes complete": ct_key, "no labels": no_layer_key})

    await page.choose_file("#paint-file", zip_path)
    if (await picture("loaded", painted_keys))[1] != large:
        failures.append(f"the loaded zip drew another picture, {out / 'loaded.png'}")

    await page.click("#paint-tool")
    await page.choose("#paint-radius", RADIUS)
    await page.choose("#paint-erase", checked=True)
    await stroke(x, y - GATED_PX)
    _, first, _ = await picture("first-erased", painted_keys)
    await stroke(x, y + GATED_PX)
    _, both, _ = await picture("both-erased", painted_keys)
    await page.choose("#paint-erase", checked=False)
    await reload(await page.address())
    if (await picture("erased-reloaded", painted_keys))[1] != both:
        failures.append(f"the reload after two erasing strokes brought back another picture, "
                        f"{out / 'erased-reloaded.png'}")
    await turn("#paint-undo")
    if (await picture("history-undone", painted_keys))[1] != first:
        failures.append(f"the undo after the reload did not give the first erasing stroke alone back, "
                        f"{out / 'history-undone.png'}")
    await turn("#paint-redo")
    if (await picture("history-redone", painted_keys))[1] != both:
        failures.append(f"the redo after the reload did not give both erasing strokes back, "
                        f"{out / 'history-redone.png'}")

    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    area = (await page.panes())["surface"]
    sx, sy = round(area["x"] + area["width"] / 2), round(area["y"] + area["height"] / 2)
    await page.mouse("mouseMoved", sx, sy)
    for _ in range(WHEELS):
        await page.mouse("mouseWheel", sx, sy, deltaX=0, deltaY=-240)
    _, surface, surface_centre = await picture("surface", {"surface": surface_key}, area, OFF_FOCUS_PX)
    await page.click("#paint-tool")
    await stroke(sx, sy)
    _, surface_painted, centre = await picture("surface-painted", {"surface": surface_key}, area, OFF_FOCUS_PX)
    if surface_painted == surface or centre[2] - centre[0] < 80 or surface_centre[2] != surface_centre[0]:
        failures.append(f"the Surface right of its centre is {centre} painted and {surface_centre} before, not label "
                        f"1's blue over grey, {out / 'surface-painted.png'}")
    await turn("#paint-undo")
    if (await picture("surface-undone", {"surface": surface_key}, area, OFF_FOCUS_PX))[1] != surface:
        failures.append(f"the Surface stroke's undo did not give the Surface back, {out / 'surface-undone.png'}")

    await page.evaluate("(() => { const frame = document.createElement('iframe'); frame.id = 'second'; "
                        "frame.src = location.href; frame.style = 'position: fixed; right: 0; bottom: 0; width: 480px; "
                        "height: 360px'; document.body.append(frame); return true })()")
    if not (note := await second("Another tab")).startswith("Another tab"):
        failures.append(f"a second page of the CT says {note!r}, not that another tab has its labels")
    if await page.evaluate(f"{SECOND_DOCUMENT}?.querySelector('#paint-here')?.hidden ?? true"):
        failures.append("the waiting second page offers no Paint here")
        return failures
    await page.evaluate(f"(() => {{ const erase = {SECOND_DOCUMENT}.querySelector('#paint-erase'); "
                        "for (const checked of [true, false]) { erase.checked = checked; "
                        "erase.dispatchEvent(new Event('change')) } return true })()")
    if not (note := await page.evaluate(SECOND_NOTE)).startswith("Another tab has"):
        failures.append(f"the waiting second page, once Erase is ticked, says {note!r}, not that another tab has its "
                        "labels")
        return failures
    await page.evaluate(RECORD_NOTES)
    await page.evaluate(f"(() => {{ const second = {SECOND_DOCUMENT}; second.querySelector('#zarr-url').value = "
                        f"{json.dumps(VOLUME)}; second.querySelector('#load-form').requestSubmit(); return true }})()")
    note = reopened(await page.wait_text("window.__secondNotes", lambda notes: reopened(notes) is not None, SECOND_S))
    if not note.startswith("Another tab has") or await page.evaluate(f"{SECOND_DOCUMENT}.querySelector('#paint-here').hidden"):
        failures.append(f"the second page, opening the CT again while it waits, says {note!r}, not that another tab has "
                        "its labels over Paint here")
        return failures
    await page.mouse("mouseMoved", sx - STROKE_PX // 2, sy)
    await page.mouse("mousePressed", sx - STROKE_PX // 2, sy, button="left", buttons=1, clickCount=1)
    for step_x in range(sx - STROKE_PX // 2 + STEP_PX, sx + 1, STEP_PX):
        await page.mouse("mouseMoved", step_x, sy, button="left", buttons=1)
    asked = time.monotonic()
    await page.evaluate(f"{SECOND_DOCUMENT}.querySelector('#paint-here').click(), true")
    note = await second("Brought back")
    print(f"the second page brought the labels back {time.monotonic() - asked:.2f} s after its Paint here: {note!r}")
    if not note.startswith("Brought back") or " and 1 edit made after them" not in note or HANDED not in note:
        failures.append(f"the second page, asking for the labels partway through a stroke here, says {note!r}, not "
                        f"that it brought them back with the stroke, the one edit past the chunks, and {HANDED!r}")
    await page.mouse("mouseReleased", sx, sy, button="left", buttons=0, clickCount=1)
    if not (note := await page.evaluate(NOTE)).startswith("Another tab took") or await page.evaluate(READY):
        failures.append(f"once the second page took the labels this one says {note!r} with its brush to be taken, not "
                        "that another tab took them with its brush put away")
    if (await page.state())["layers"]:
        failures.append("once the second page took the labels this one still draws them")
    await page.evaluate("(document.querySelector('#second').style.visibility = 'hidden', true)")
    await page.click("#paint-here")
    await page.browser.wait_for(READY, TIMEOUT_S)
    note = await page.evaluate(NOTE)
    if not note.startswith("Brought back") or HANDED not in note:
        failures.append(f"taking the labels back from the idle second page, this one says {note!r}, not that it "
                        f"brought them back with {HANDED!r}")
    if not (note := await second("Another tab took")).startswith("Another tab took"):
        failures.append(f"once this page took the labels back the second says {note!r}, not that another tab took them")
    handed_back = (await picture("handed-back", {"surface": surface_key}, area, OFF_FOCUS_PX))[1]
    await turn("#paint-undo")
    handed_undone = (await picture("handed-undone", {"surface": surface_key}, area, OFF_FOCUS_PX))[1]
    if handed_back == surface or handed_undone != surface:
        failures.append(f"the labels back from the second page lack the stroke it ended, or its undo does not give "
                        f"the Surface back, {out / 'handed-back.png'}, {out / 'handed-undone.png'}")
    await page.click("#paint-tool")
    await page.choose("#paint-radius", LARGE_RADIUS)
    for _ in range(BACKLOG_WHEELS):
        await page.mouse("mouseWheel", sx, sy, deltaX=0, deltaY=240)
    await page.settle(time.monotonic(), {"surface": surface_key})
    for offset in BACKLOG_PX:
        await stroke(sx, sy + offset, lambda: asyncio.sleep(0))
    await kept()
    await page.choose_file("#paint-file", zip_path)
    await page.evaluate("(document.querySelector('#second').style.visibility = 'visible', true)")
    await page.reveal("#open-source-button")
    await page.click("#open-source-button")
    await page.type_into("zarr-url", OTHER)
    await page.click("#load-form button[type='submit']")
    note = await second("Brought back")
    await page.evaluate(f"(window.__secondBefore = {SECOND_DOCUMENT}, {SECOND_DOCUMENT}.defaultView.location.reload(), "
                        "true)")
    await page.wait_text(f"{SECOND_DOCUMENT} !== window.__secondBefore && {SECOND_NOTE}",
                         lambda note: note and note.startswith("Brought back"), SECOND_S)
    reloaded = await page.evaluate(SECOND_NOTE)
    print(f"once this page opened another volume during a load the second said {note!r}, and after its reload "
          f"{reloaded!r}")
    if BACKLOGGED not in note or BACKLOGGED not in reloaded:
        failures.append(f"once this page opened another volume during a load the second said {note!r}, and after its "
                        f"reload {reloaded!r}, not that it brought back the labels with {BACKLOGGED!r}")
    return failures
