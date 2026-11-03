import asyncio
import base64
import json
import re
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, POLL_S, QUIET_S, TIMEOUT_S, VOLUME, ct_key, lines_key, surface_key

ZOOM = 4
SPACING = 5
CLICKS_PX = [(-40, -20), (40, 20)]
STROKE_PX = (-60, 60, 60)
STEP_PX = 10
CT = "20260411134726-2.400um-0.2m-78keV-masked"
SAVED = f"{CT}-points.zip"
LEVELS = 6
FILES = [f"{CT}.nml", *(f"{CT}-edited-level-{level}.nml" for level in range(1, LEVELS))]
COORDINATES = re.compile(r' x="([^"]*)" y="([^"]*)" z="([^"]*)"')
KEPT = ", kept in this browser."
NOTE = "document.querySelector('#points-note').textContent"
COARSER = "is in a coarser level's voxels"
PIXELS = """(async ([png, points]) => {
  const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
    { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  return points.map(([x, y]) => [...context.getImageData(x, y, 1, 1).data.slice(0, 3)]);
})"""


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    if (place := await page.place("#points-section")) != "Points":
        failures.append(f"its card is on {place!r}, not a Points tab of its own")
    await page.reveal("#points-section")
    await page.choose("#points-follow", checked=False)
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", str(ZOOM))
    await page.settle(t0, {"panes complete": ct_key})
    rect = (await page.panes())["xy"]
    x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
    voxels_per_px = await page.config("BASE_VIEW_WIDTH_VOXELS") / ZOOM / min(rect["width"], rect["height"])
    centre_XYZ = [float(value) for value in CENTER_XYZ.split(",")]
    clicks = [(x + dx, y + dy) for dx, dy in CLICKS_PX]

    async def picture(name, area=rect, keys={"panes complete": ct_key}):
        await page.mouse("mouseMoved", 150, 950)
        await page.settle(time.monotonic(), keys)
        png, pixels = await page.picture(area)
        (out / f"{name}.png").write_bytes(png)
        at = [base64.b64encode(png).decode(), [[cx - round(area["x"]), cy - round(area["y"])] for cx, cy in clicks]]
        return pixels["sha256"], await page.evaluate(f"{PIXELS}({json.dumps(at)})")

    async def click_all():
        for click_x, click_y in clicks:
            await page.mouse("mouseMoved", click_x, click_y)
            await page.mouse("mousePressed", click_x, click_y, button="left", buttons=1, clickCount=1)
            await page.mouse("mouseReleased", click_x, click_y, button="left", buttons=0, clickCount=1)
        await kept()

    async def kept():
        await page.wait_text(NOTE, lambda note: note.endswith(KEPT), 60)

    async def stroke():
        from_x, to_x, stroke_y = x + STROKE_PX[0], x + STROKE_PX[1], y + STROKE_PX[2]
        await page.mouse("mouseMoved", from_x, stroke_y)
        await page.mouse("mousePressed", from_x, stroke_y, button="left", buttons=1, clickCount=1)
        for step_x in range(from_x + STEP_PX, to_x + 1, STEP_PX):
            await page.mouse("mouseMoved", step_x, stroke_y, button="left", buttons=1)
        await page.mouse("mouseReleased", to_x, stroke_y, button="left", buttons=0, clickCount=1)
        await kept()

    ct, _ = await picture("ct")
    await page.click("#points-tool")
    await page.choose("#points-joined", "connected")
    await click_all()
    clicked, (first, second) = await picture("clicked")
    if first[2] - first[0] < 80 or second != [255, 255, 255]:
        failures.append(f"the first node is {first} and the selected second {second}, not fiber 1's blue and white")

    await stroke()
    stroked, _ = await picture("stroked")
    await page.click("#points-undo")
    undone, _ = await picture("undone")
    if undone != clicked:
        failures.append(f"the undo drew another picture than before the stroke, {out / 'undone.png'}")
    await page.click("#points-redo")
    redone, _ = await picture("redone")
    if redone != stroked or stroked == clicked:
        failures.append(f"the redo drew another picture than the stroke, {out / 'redone.png'}")
    await kept()
    note = await page.evaluate(NOTE)

    await page.open(await page.address())
    restored, _ = await picture("restored")
    brought = await page.evaluate(NOTE)
    count = note.split(" nodes")[0].split(", ")[-1]
    if brought != f"Brought back {count} nodes placed here before.":
        failures.append(f"the reload says {brought!r} of the {note!r} kept")
    if restored != stroked:
        failures.append(f"the reload brought back another picture, {out / 'restored.png'}")

    page.browser.downloads.mkdir(parents=True, exist_ok=True)
    before = set(page.browser.downloads.iterdir())
    zip_path = await page.download("#points-save", SAVED)
    await asyncio.sleep(QUIET_S)
    (out / SAVED).write_bytes(zip_path.read_bytes())
    if (downloads := sorted(path.name for path in set(page.browser.downloads.iterdir()) - before)) != [SAVED]:
        failures.append(f"Save downloaded {downloads}, not one {SAVED}")
    with zipfile.ZipFile(zip_path) as saved:
        files = {name: saved.read(name).decode() for name in saved.namelist()}
        stored = [info.filename for info in saved.infolist() if info.compress_type != zipfile.ZIP_DEFLATED]
    if stored:
        failures.append(f"the zip holds {stored} as they are, not deflated")
    if list(files) != FILES:
        failures.append(f"the zip holds {list(files)}, not {FILES}")
        return failures
    finest = files[FILES[0]]
    # Each coarser level's file is the finest's but for its coordinates: its voxel v is centred at finest voxel
    # v * downsample + (downsample - 1) / 2.
    for level, name in enumerate(FILES[1:], 1):
        downsample = 2 ** level
        moved = [(float(value) - (downsample - 1) / 2) / downsample for match in COORDINATES.findall(finest) for value in match]
        read = [float(value) for match in COORDINATES.findall(files[name]) for value in match]
        same = COORDINATES.sub("", files[name]) == COORDINATES.sub("", finest) and len(read) == len(moved)
        if not same or any(abs(a - b) > 1e-3 for a, b in zip(read, moved)):
            failures.append(f"{name} is not the finest NML's nodes at level {level}'s voxels, {out / SAVED}")
    nml_path = page.browser.downloads / FILES[0]
    nml_path.write_text(finest)
    things = ElementTree.fromstring(finest).findall("thing")
    nodes = [node.attrib for thing in things for node in thing.iter("node")]
    xyz = [[float(node[axis]) for axis in "xyz"] for node in nodes]
    placed = [[centre_XYZ[0] + dx * voxels_per_px, centre_XYZ[1] - dy * voxels_per_px, centre_XYZ[2]] for dx, dy in CLICKS_PX]
    placed.append([centre_XYZ[0] + STROKE_PX[0] * voxels_per_px, centre_XYZ[1] - STROKE_PX[2] * voxels_per_px, centre_XYZ[2]])
    edges = sum(len(thing.findall("edges/edge")) for thing in things)
    if len(things) != 1 or len(nodes) < 3 or edges != len(nodes) - 1:
        failures.append(f"the NML holds {len(things)} trees of {len(nodes)} nodes and {edges} edges, not one chain")
    elif any(max(abs(a - b) for a, b in zip(got, want)) > 1 for got, want in zip(xyz, placed)):
        failures.append(f"the NML's first nodes {xyz[:3]} are more than a voxel from where they were placed, {placed}")
    else:
        stroke_XYZ = xyz[2:]
        steps = [sum((b - a) ** 2 for a, b in zip(p, q)) ** 0.5 for p, q in zip(stroke_XYZ, stroke_XYZ[1:])]
        length = (STROKE_PX[1] - STROKE_PX[0]) * voxels_per_px
        if any(abs(step - SPACING) > 1e-3 for step in steps) or len(stroke_XYZ) != 1 + int(length // SPACING):
            failures.append(f"the stroke's {len(stroke_XYZ)} nodes are {steps} apart, not every {SPACING} over {length:.1f} voxels")
        if len({node.get("teca.strokeId") for node in nodes[2:]}) != 1 or "teca.strokeId" in nodes[0]:
            failures.append(f"the stroke ids are {[node.get('teca.strokeId') for node in nodes]}, not one for the stroke alone")

    await page.click("#points-clear")
    cleared, _ = await picture("cleared")
    if len(page.browser.dialogs) != 1:
        failures.append(f"Clear all asked {page.browser.dialogs}, not once")
    if cleared != ct:
        failures.append(f"clearing did not give xy back, {out / 'cleared.png'}")
    await page.open(await page.address())
    reloaded, _ = await picture("cleared-reloaded")
    if reloaded != ct:
        failures.append(f"the reload after clearing brought marks back, {out / 'cleared-reloaded.png'}")

    await page.choose_file("#points-file", zip_path)
    loaded, _ = await picture("loaded")
    if loaded != stroked:
        failures.append(f"the zip opened drew another picture, {out / 'loaded.png'}")
    await page.click("#points-clear")
    await picture("cleared-again")
    await page.choose_file("#points-file", nml_path)
    loaded, _ = await picture("loaded-nml")
    if loaded != stroked:
        failures.append(f"the NML opened drew another picture, {out / 'loaded-nml.png'}")
    level_path = page.browser.downloads / FILES[1]
    level_path.write_text(files[FILES[1]])
    asked = len(page.browser.dialogs)
    await page.choose_file("#points-file", level_path)
    refused = await page.wait_text(NOTE, lambda note: COARSER in note or len(page.browser.dialogs) > asked, 60)
    unchanged, _ = await picture("level-refused")
    if not refused.startswith(f"{FILES[1]} {COARSER}") or page.expected(COARSER) != 1 or unchanged != stroked:
        failures.append(f"Open… of {FILES[1]} alone says {refused!r} and draws {out / 'level-refused.png'}, not "
                        "its refusal over the trees there were")

    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    area = (await page.panes())["surface"]
    x, y = round(area["x"] + area["width"] / 2), round(area["y"] + area["height"] / 2)
    clicks = [(x + dx, y + dy) for dx, dy in CLICKS_PX]
    surface, _ = await picture("surface", area, {"surface": surface_key})
    await page.click("#points-tool")
    await click_all()
    t0 = time.monotonic()
    while True:
        _, (first, second) = await picture("surface-clicked", area, {"surface": surface_key})
        if first[2] - first[0] >= 80 and second == [255, 255, 255] or time.monotonic() - t0 > TIMEOUT_S:
            break
        await asyncio.sleep(POLL_S)
    if first[2] - first[0] < 80 or second != [255, 255, 255]:
        failures.append(f"the Surface's first node is {first} and the selected second {second}, not fiber 1's blue "
                        f"and white, {out / 'surface-clicked.png'}")
    for _ in clicks:
        await page.click("#points-undo")
    undone, _ = await picture("surface-undone", area, {"surface": surface_key})
    if undone != surface:
        failures.append(f"taking the Surface's points back did not give it back, {out / 'surface-undone.png'}")
    return failures
