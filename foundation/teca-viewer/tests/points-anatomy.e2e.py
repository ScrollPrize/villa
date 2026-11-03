import re
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, NEUTRAL_XY, VOLUME, ct_key

ZOOM = 4
STEP_PX = 10
FIBER_PX = (-90, -20, -40)
INK_PX = (-90, -20, 40)
PAST_GAP_PX = (20, 90, -40)
NEXT_WINDING_PX = (-90, 90, 90)
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser."
EXPECTED = {
    "S0_W0_HZ_RECTO_FIBER_1": (["W0", "recto"], ("S0", "W0", "HZ", "RECTO", "FIBER", None)),
    "S0_W0_RECTO_SURFACE_INK": (["W0", "recto"], ("S0", "W0", None, "RECTO", "SURFACE", "INK")),
    "S0_W1_HZ_RECTO_FIBER_1": (["W+1", "recto"], ("S0", "W1", "HZ", "RECTO", "FIBER", None)),
}


def classification(name):
    found = {"surface": None, "winding": None, "orientation": None, "side": None, "type": None, "ink": None}
    for part in filter(None, (part.strip() for part in name.split("_"))):
        upper = part.upper()
        if re.fullmatch(r"S(?:N|-?\d+)", upper):
            found["surface"] = upper
        elif re.fullmatch(r"W(?:N|-?\d+)", upper):
            found["winding"] = upper
        else:
            for field, tags in [("orientation", ["HZ", "VT"]), ("side", ["RECTO", "VERSO"]),
                                ("type", ["FIBER", "SURFACE"]), ("ink", ["INK", "NOINK"])]:
                if upper in tags:
                    found[field] = upper
    if found["type"] is None and found["orientation"] is not None:
        found["type"] = "FIBER"
    if found["type"] != "FIBER":
        found["orientation"] = None
    return tuple(found.values())


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.reveal("#points-section")
    await page.choose("#points-follow", checked=False)
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", str(ZOOM))
    await page.settle(t0, {"panes complete": ct_key})
    rect = (await page.panes())["xy"]
    x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)

    async def picture(name):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"panes complete": ct_key})
        png, pixels = await page.picture(rect)
        (out / f"{name}.png").write_bytes(png)
        return pixels["sha256"]

    async def note():
        return await page.evaluate("document.querySelector('#points-note').textContent")

    async def target():
        return await page.evaluate("(() => { const e = document.querySelector('#tool-label'); "
                                   "return e.hidden ? null : e.textContent })()")

    async def stroke(span, during=None):
        from_x, to_x, stroke_y = x + span[0], x + span[1], y + span[2]
        await page.mouse("mouseMoved", from_x, stroke_y)
        await page.mouse("mousePressed", from_x, stroke_y, button="left", buttons=1, clickCount=1)
        for step_x in range(from_x + STEP_PX, to_x + 1, STEP_PX):
            await page.mouse("mouseMoved", step_x, stroke_y, button="left", buttons=1)
            if during is not None and step_x >= (from_x + to_x) / 2:
                await during()
                during = None
        await page.mouse("mouseReleased", to_x, stroke_y, button="left", buttons=0, clickCount=1)
        await page.wait_text("document.querySelector('#points-note').textContent", lambda note: note.endswith(KEPT), 60)

    labels = []
    await page.click("#points-tool")
    await page.key("K", shift=True)
    await stroke(FIBER_PX)
    await page.key("Escape")
    await page.mouse("mouseMoved", x + FIBER_PX[1], y + FIBER_PX[2])
    await page.key("a")
    labels.append(await target())
    await stroke(INK_PX)
    await page.key("Tab")
    labels.append(await target())
    await stroke(PAST_GAP_PX)
    await page.key("]")
    labels.append(await target())
    if labels != ["W0 › recto › ink", "W0 › recto › fiber 1", "W+1 › recto › fiber 1"]:
        failures.append(f"after A, Tab and ] the label beside the still pointer showed {labels}, not W0 › recto › ink, "
                        "W0 › recto › fiber 1 and W+1 › recto › fiber 1")
    moved = []

    async def keys_during_drag():
        await page.key("Tab")
        await page.key("c")
        moved.append(await target())

    await stroke(NEXT_WINDING_PX, keys_during_drag)
    if moved != ["W+1 › recto › fiber 1"] or await target() != "W+1 › recto › fiber 1":
        failures.append(f"Tab during the drag moved the class to {moved}, then {await target()!r}")
    drawn = await picture("drawn")
    kept_note = await note()

    await page.open(await page.address())
    restored = await picture("restored")
    count = kept_note.split(" nodes")[0].split(", ")[-1]
    if await note() != f"Brought back {count} nodes placed here before.":
        failures.append(f"the reload says {await note()!r} of the {kept_note!r} kept")
    if restored != drawn:
        failures.append(f"the reload drew another picture, {out / 'restored.png'} against {out / 'drawn.png'}")

    async def saved():
        path = await page.download("#points-save", SAVED)
        (out / SAVED).write_bytes(path.read_bytes())
        with zipfile.ZipFile(path) as zipped:
            finest = zipped.read(FINEST)
        path.unlink()
        return finest

    finest = await saved()
    root = ElementTree.fromstring(finest)
    paths = {}

    def walk(element, path):
        for group in element.findall("group"):
            paths[group.get("id")] = [*path, group.get("name")]
            walk(group, paths[group.get("id")])

    walk(root.find("groups"), [])
    if sorted(map(tuple, paths.values())) != [("W+1",), ("W+1", "recto"), ("W0",), ("W0", "recto")]:
        failures.append(f"the NML's groups are {sorted(paths.values())}, not W0 and W+1 each holding a recto")
    things = {thing.get("name"): thing for thing in root.findall("thing")}
    if sorted(things) != sorted(EXPECTED):
        failures.append(f"the NML's trees are {sorted(things)}, not {sorted(EXPECTED)}")
    for name, (path, parsed) in EXPECTED.items():
        if name in things and paths.get(things[name].get("groupId")) != path:
            failures.append(f"{name} is in {paths.get(things[name].get('groupId'))}, not {path}")
        if classification(name) != parsed:
            failures.append(f"parseTreeClassification reads {name} as {classification(name)}, not {parsed}")
    fiber = things.get("S0_W0_HZ_RECTO_FIBER_1")
    if fiber is not None:
        nodes = sorted((int(node.get("id")), node.get("teca.strokeId")) for node in fiber.iter("node"))
        strokes = list(dict.fromkeys(stroke_id for _, stroke_id in nodes))
        if len(strokes) != 2 or None in strokes or fiber.findall("edges/edge"):
            failures.append(f"fiber 1 has scribbles {strokes} and {len(fiber.findall('edges/edge'))} edges, not two and none")
        first = [node_id for node_id, stroke_id in nodes if stroke_id == strokes[0]]
        comments = {comment.get("node"): comment.get("content") for comment in root.iter("comment")}
        if comments != {str(node_id): "[KOLLESIS]" for node_id in first}:
            failures.append(f"the comments are {comments}, not [KOLLESIS] on each node of the first scribble, {first}")

    await page.click("#points-clear")
    opened_zip = page.browser.downloads / "opened.zip"
    opened_zip.write_bytes((out / SAVED).read_bytes())
    await page.choose_file("#points-file", opened_zip)
    opened_note = await page.wait_text("document.querySelector('#points-note').textContent", lambda note: note.endswith(KEPT), 60)
    if opened_note.split(" nodes")[0].split(", ")[-1] != count:
        failures.append(f"the zip opened says {opened_note!r} of the {kept_note!r} saved")
    if await picture("opened") != drawn:
        failures.append(f"the zip opened drew another picture, {out / 'opened.png'} against {out / 'drawn.png'}")
    if await saved() != finest:
        failures.append(f"the zip opened saves another NML than it holds, {out / SAVED}")
    return failures
