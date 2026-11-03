import base64
import json
import re
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, NEUTRAL_XY, VOLUME, ct_key

ZOOM = 4
STEP_PX = 10
FIRST_PX = (60, -40)
SECOND_PX = (-60, 40)
STROKES = [(-120, "K", "kollesis"), (-80, "K", None), (80, "Escape", None), (120, "K", "kollesis")]
NEXT_WINDING_ROW_PX = 160
AROUND_NODE_PX = [(4, 3), (-4, 3), (4, -3), (-4, -3), (0, 7), (0, -7)]
BETWEEN_PX = [(0, 6), (0, -6)]
RING_PX = [(0, 3), (0, -3)]
CROWDED_ROW_PX = 100
CROWDED_SPACING = "1"
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser."
NOTE = "document.querySelector('#points-note').textContent"
FIBER = "W0 › recto › fiber 1"
NEXT_FIBER = "W+1 › recto › fiber 1"
FIBER_TREE = "S0_W0_HZ_RECTO_FIBER_1"
NEXT_FIBER_TREE = "S0_W1_HZ_RECTO_FIBER_1"
TAGS = {"kollesis": "[KOLLESIS]", None: None}
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
    await page.reveal("#points-section")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", str(ZOOM))
    await page.settle(t0, {"panes complete": ct_key})
    rect = (await page.panes())["xy"]
    x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
    voxels_per_px = await page.config("BASE_VIEW_WIDTH_VOXELS") / ZOOM / min(rect["width"], rect["height"])

    async def read(js):
        return await page.evaluate(js)

    async def focus():
        return [float(value) for value in (await read("document.querySelector('#position-number').value")).split(",")]

    async def kept():
        await page.wait_text(NOTE, lambda note: note.endswith(KEPT), 60)

    async def label():
        return await read("(() => { const e = document.querySelector('#tool-label'); return e.hidden ? null : e.textContent })()")

    async def click(dx, dy):
        await page.mouse("mouseMoved", x + dx, y + dy)
        await page.mouse("mousePressed", x + dx, y + dy, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", x + dx, y + dy, button="left", buttons=0, clickCount=1)
        await kept()

    async def stroke(row):
        await page.mouse("mouseMoved", x - 90, y + row)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1)
        for step_x in range(x - 90 + STEP_PX, x + 91, STEP_PX):
            await page.mouse("mouseMoved", step_x, y + row, button="left", buttons=1)
        await page.mouse("mouseReleased", x + 90, y + row, button="left", buttons=0, clickCount=1)
        await kept()

    async def saved(kept=None):
        path = await page.download("#points-save", SAVED)
        if kept is not None:
            kept.write_bytes(path.read_bytes())
        with zipfile.ZipFile(path) as zipped:
            root = ElementTree.fromstring(zipped.read(FINEST))
        path.unlink()
        return root

    async def held(name, shift=False):
        await page.key(name, down_only=True, shift=shift)
        await page.key(name, repeat=True, shift=shift)

    def off(got, want):
        return max(abs(a - b) for a, b in zip(got, want))

    if not await read("document.querySelector('#points-follow').checked"):
        failures.append("Follow is off at first")
    await held("6")
    if not await read("document.querySelector('#points-tool').classList.contains('active')"):
        failures.append("6 held left the pen put away")
        return failures
    centre_XYZ = await focus()
    await click(*FIRST_PX)
    first_XYZ = [centre_XYZ[0] + FIRST_PX[0] * voxels_per_px, centre_XYZ[1] - FIRST_PX[1] * voxels_per_px, centre_XYZ[2]]
    if off(followed := await focus(), first_XYZ) > 1:
        failures.append(f"the click with Follow on left the focus at {followed}, not at its node {first_XYZ}")
    await held("O", shift=True)
    if await read("document.querySelector('#points-follow').checked"):
        failures.append("Shift+O left Follow on")
    await click(*SECOND_PX)
    if (unfollowed := await focus()) != followed:
        failures.append(f"the click with Follow off moved the focus from {followed} to {unfollowed}")
    await held("B")
    root = await saved()
    flagged = [branchpoint.get("id") for branchpoint in root.iter("branchpoint")]
    if flagged != [second := root.find("parameters/activeNode").get("id")]:
        failures.append(f"B held after the second click left branchpoints {flagged}, not its node {second}")
    await page.settle(time.monotonic(), {"panes complete": ct_key})

    async def pressed(selector):
        return await read(f"[...document.querySelectorAll({json.dumps(selector)})]"
                          ".filter((button) => button.getAttribute('aria-pressed') === 'true').map((button) => button.textContent.toLowerCase())")

    labels = []
    for row, key, _ in STROKES:
        await page.mouse("mouseMoved", x - 90, y + row)
        await held(key, shift=key != "Escape")
        labels.append([await label(), await pressed("#points-mark-kollesis")])
        await stroke(row)
    await page.key("]")
    labels.append([await label(), await pressed("#points-mark-kollesis")])
    await stroke(NEXT_WINDING_ROW_PX)
    for _ in range(2):
        await held("Tab")
        labels.append([await label(), await pressed("#points-mark-kollesis")])
    want = [[FIBER + (f" › {mark}" if mark else ""), [mark] if mark else []] for _, _, mark in STROKES]
    want += [[f"{fiber} › kollesis", ["kollesis"]] for fiber in [NEXT_FIBER, FIBER, NEXT_FIBER]]
    if labels != want:
        failures.append(f"after each key the label beside the pointer and Mark's pressed buttons were {labels}, not {want}")
        return failures

    async def runs():
        return await read("[...document.querySelectorAll('#points-runs button')].map((button) => [...button.children].map((span) => span.textContent))")

    listed = await runs()
    await held("w")
    only = await runs()

    async def count():
        return int(re.search(r"(\d+) nodes?\b", await read(NOTE))[1])

    before = await count()
    await click(-90, STROKES[0][0])
    if (placed := [await count(), await label()]) != [before + 1, f"{NEXT_FIBER} › kollesis"]:
        failures.append(f"with W on a click on a node W hides gives [the nodes, the label] {placed}, not a node "
                        f"placed, {before + 1}, in {NEXT_FIBER}")
    else:
        await page.click("#points-undo")
        await kept()
    await held("w")
    if await runs() != listed or len(listed) != 3 or only != listed[2:]:
        failures.append(f"the runs listed are {listed}, and {only} with W on, not three and W+1's alone")

    async def step(how):
        if how == "first":
            await page.click('#points-runs button[data-run="0"]')
        else:
            await page.key(how)
        return await focus(), await read("document.querySelector('#points-note').textContent")

    steps = [await step(how) for how in ["first", ".", ".", ".", ","]]
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    png, _ = await page.picture(rect)
    (out / "banded.png").write_bytes(png)

    root = await saved(out / SAVED)
    comments = {comment.get("node"): comment.get("content") for comment in root.iter("comment")}
    want_rows = {0: (FIBER_TREE, None), SECOND_PX[1]: (FIBER_TREE, None)}
    want_rows |= {row: (FIBER_TREE, TAGS[mark]) for row, _, mark in STROKES}
    want_rows[NEXT_WINDING_ROW_PX] = (NEXT_FIBER_TREE, TAGS["kollesis"])
    rows = {}
    for thing in root.findall("thing"):
        for node in thing.iter("node"):
            point = [float(node.get(axis)) for axis in "xyz"]
            names, tags, nodes = rows.setdefault(round((followed[1] - point[1]) / voxels_per_px), (set(), set(), []))
            names.add(thing.get("name"))
            tags.add(comments.get(node.get("id")))
            nodes.append([int(node.get("id")), *point])
    got_rows = {row: (names, tags) for row, (names, tags, _) in rows.items()}
    if got_rows != {row: ({name}, {tag}) for row, (name, tag) in want_rows.items()}:
        failures.append(f"the NML's rows hold trees and tags {got_rows}, not {want_rows}")
        return failures

    marked = [sorted(rows[row][2]) for row in [STROKES[0][0], STROKES[3][0], NEXT_WINDING_ROW_PX]]
    names = [f"{FIBER} › kollesis", f"{FIBER} › kollesis", f"{NEXT_FIBER} › kollesis"]
    want = [[name, f"{len(nodes)} nodes"] for name, nodes in zip(names, marked)]
    if listed != want:
        failures.append(f"the runs listed are {listed}, not {want}")
    middle = [nodes[len(nodes) // 2][1:] for nodes in marked]
    stepped = [(0, "the run clicked"), (1, "."), (2, "."), (2, ". past the last"), (1, ",")]
    for (focused, _), (run, name) in zip(steps, stepped):
        if off(focused, middle[run]) > 1:
            failures.append(f"after {name} the focus is {focused}, not run {run}'s middle {middle[run]}")
    if steps[3][1] != "No marked run lies that way.":
        failures.append(f". past the last run says {steps[3][1]!r}")
    if (active := root.find("parameters/activeNode").get("id")) != str(marked[1][0][0]):
        failures.append(f"the selected node is {active}, not the first of the second run, {marked[1][0][0]}")

    middle_node, next_node = marked[1][len(marked[1]) // 2], marked[1][len(marked[1]) // 2 + 1]
    centred = steps[-1][0]

    def screen(node, dx=0, dy=0):
        return [round(x + (node[1] - centred[0]) / voxels_per_px + dx - rect["x"]),
                round(y - (node[2] - centred[1]) / voxels_per_px + dy - rect["y"])]

    between = [(a + b) / 2 for a, b in zip(middle_node, next_node)]
    places = [screen(middle_node), screen(between)] + [screen(middle_node, *px) for px in AROUND_NODE_PX]
    places += [screen(between, *px) for px in BETWEEN_PX]
    places += [screen(middle_node, *px) for px in RING_PX]
    node_rgb, line_rgb, *band = await read(f"{PIXELS}({json.dumps([base64.b64encode(png).decode(), places])})")
    band, ring = band[:-len(RING_PX)], band[-len(RING_PX):]
    if not all(max(rgb) < 40 for rgb in ring):
        failures.append(f"the band's middle node, {5 / voxels_per_px:.1f} px from the next, has no disc: {ring}")
    gold = [r > 200 and 130 < g < 230 and b < 90 for r, g, b in [node_rgb, line_rgb, *band]]
    if gold[0] or gold[1]:
        failures.append(f"the node and its line are {node_rgb} and {line_rgb}, the band's gold over them, {out / 'banded.png'}")
    if not all(gold[2:]):
        failures.append(f"about the node and between it and the next the band is {band}, not all gold, {out / 'banded.png'}")
    await page.mouse("mouseMoved", x, y)
    await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
    await page.mouse("mouseReleased", x, y, button="left", buttons=0, clickCount=1)
    root = await saved()
    if (active := root.find("parameters/activeNode").get("id")) != str(middle_node[0]):
        failures.append(f"the click on the band's middle node selected {active}, not {middle_node[0]}")

    async def rings(png, node):
        return await read(f"{PIXELS}({json.dumps([base64.b64encode(png).decode(), [screen(node, *px) for px in RING_PX]])})")

    await page.key("Escape")
    await page.choose("#points-spacing", CROWDED_SPACING)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    blank, _ = await page.picture(rect)
    await page.mouse("mouseMoved", x - 90, y + CROWDED_ROW_PX)
    await page.mouse("mousePressed", x - 90, y + CROWDED_ROW_PX, button="left", buttons=1, clickCount=1)
    for step_x in range(x - 90 + STEP_PX, x + 91, STEP_PX):
        await page.mouse("mouseMoved", step_x, y + CROWDED_ROW_PX, button="left", buttons=1)
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    held_png, _ = await page.picture(rect)
    await page.mouse("mouseReleased", x + 90, y + CROWDED_ROW_PX, button="left", buttons=0, clickCount=1)
    await kept()
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    released_png, _ = await page.picture(rect)
    root = await saved()
    crowded = sorted([int(node.get("id")), *[float(node.get(axis)) for axis in "xyz"]] for node in root.iter("node")
                     if round((followed[1] - float(node.get("y"))) / voxels_per_px) == STROKES[3][0] + CROWDED_ROW_PX)
    (out / "crowded-pressed.png").write_bytes(held_png)
    (out / "crowded.png").write_bytes(released_png)
    if len(crowded) < 3 or (apart := abs(crowded[1][1] - crowded[0][1]) / voxels_per_px) >= 8:
        failures.append(f"the stroke at spacing {CROWDED_SPACING} placed {len(crowded)} nodes, not crowded ones")
        return failures
    node = crowded[len(crowded) // 2]
    empty = await rings(blank, node)
    near = lambda a, b: max(abs(p - q) for p, q in zip(a, b)) <= 8
    if not all(max(rgb) < 40 for rgb in await rings(held_png, node)):
        failures.append(f"a pressed stroke's node, {apart:.1f} px from the next, has no disc, {out / 'crowded-pressed.png'}")
    if not all(near(a, b) for a, b in zip(await rings(released_png, node), empty)):
        failures.append(f"a node {apart:.1f} px from the next has its disc once released, {out / 'crowded.png'}")
    at_x, at_y = [rect[axis] + value for axis, value in zip("xy", screen(node))]
    await page.mouse("mouseMoved", at_x, at_y)
    await page.mouse("mousePressed", at_x, at_y, button="left", buttons=1, clickCount=1)
    await page.mouse("mouseReleased", at_x, at_y, button="left", buttons=0, clickCount=1)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    selected_png, _ = await page.picture(rect)
    (out / "crowded-selected.png").write_bytes(selected_png)
    root = await saved()
    if (active := root.find("parameters/activeNode").get("id")) != str(node[0]):
        failures.append(f"the click on crowded node {node[0]}, its disc left out, selected {active}")
    elif not all(max(rgb) < 40 for rgb in await rings(selected_png, node)):
        failures.append(f"the selected crowded node {node[0]} has no disc, {out / 'crowded-selected.png'}")
    return failures
