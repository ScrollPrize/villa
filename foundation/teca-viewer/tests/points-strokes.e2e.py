import re
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, VOLUME, ct_key

ZOOM = 4
STEP_PX = 10
FOLLOWED_ROWS_PX = [-60, 40, 40]
CLICK_PX = (60, -70)
FIRST_ROW_PX, SECOND_ROW_PX = -30, 40
SECOND_TREE_ROW_PX = -60
FIBER, SECOND_FIBER = "fiber 1", "fiber 2"
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser."
SCRIBBLE_ROWS_PX = [-60, -20, 20, 60]
OFF_PX = 100
FIBER_1, FIBER_2, NO_INK = "S0_W0_HZ_RECTO_FIBER_1", "S0_W0_HZ_RECTO_FIBER_2", "S0_W0_RECTO_SURFACE_NOINK"
LAST = "W0 › recto › fiber 1 has no scribble after this one."
SCRIBBLE = ("[document.querySelector('#points-scribble-place')?.textContent ?? null, "
            "document.querySelector('#points-scribble-count').textContent]")


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

    async def read(js):
        return await page.evaluate(js)

    async def count():
        await page.wait_text("document.querySelector('#points-note').textContent", lambda note: note.endswith(KEPT), 60)
        return int(re.search(r"(\d+) nodes", await read("document.querySelector('#points-note').textContent"))[1])

    async def stroke():
        return await read("document.querySelector('#points-scribble-count').textContent")

    async def target():
        return await read("[document.querySelector('#points-fiber').textContent, "
                          "document.querySelector('#points-class').options.length, "
                          "document.querySelector('#points-scribble-count').textContent]")

    async def focus():
        return [float(value) for value in (await read("document.querySelector('#position-number').value")).split(",")]

    async def saved():
        path = await page.download("#points-save", SAVED)
        with zipfile.ZipFile(path) as zipped:
            root = ElementTree.fromstring(zipped.read(FINEST))
        path.unlink()
        strokes = {}
        for node in root.iter("node"):
            point = [float(node.get(axis)) for axis in "xyz"]
            strokes.setdefault(node.get("teca.strokeId") or node.get("id"), []).append([int(node.get("id")), *point])
        active = root.find("parameters/activeNode")
        classes = {thing.get("name"): sorted(int(node.get("id")) for node in thing.iter("node"))
                   for thing in root.findall("thing")}
        return (sorted(sorted(nodes) for nodes in strokes.values()), None if active is None else int(active.get("id")),
                classes)

    def ids(*strokes):
        return sorted(node[0] for stroke in strokes for node in stroke)

    async def centred_on(stroke):
        return max(abs(a - b) for a, b in zip(await focus(), stroke[len(stroke) // 2][1:])) <= 1

    def scribbled(classes, strokes):
        return {name: [n for n, stroke in enumerate(strokes, 1) if set(ids(stroke)) & set(held)]
                for name, held in classes.items()}

    async def click(dx, dy):
        await page.mouse("mouseMoved", x + dx, y + dy)
        await page.mouse("mousePressed", x + dx, y + dy, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", x + dx, y + dy, button="left", buttons=0, clickCount=1)

    async def drag(row):
        await page.mouse("mouseMoved", x - 90, y + row)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1)
        for step_x in range(x - 90 + STEP_PX, x + 91, STEP_PX):
            await page.mouse("mouseMoved", step_x, y + row, button="left", buttons=1)
        await page.mouse("mouseReleased", x + 90, y + row, button="left", buttons=0, clickCount=1)

    await page.click("#points-tool")
    for index, row in enumerate(FOLLOWED_ROWS_PX):
        await page.choose("#points-joined", "connected" if index == 2 else "separate")
        await drag(row)
        await count()
    await page.choose("#points-joined", "separate")
    strokes, _, _ = await saved()
    if len(strokes) != 3:
        failures.append(f"the three strokes drawn with Follow on are {len(strokes)}")
        return failures
    one, two, three = strokes
    for _ in three:
        await page.key(";")
    if (selected := (await saved())[1]) != two[-1][0]:
        failures.append(f"; pressed {len(three)} times selected node {selected}, not the second stroke's last")
        return failures
    before = await focus()
    await page.key("Delete")
    await count()
    selected, after = (await saved())[1], await focus()
    if selected != one[-1][0]:
        failures.append(f"deleting the second stroke selected node {selected}, not the first stroke's last, {one[-1][0]}")
    if after != before:
        failures.append(f"deleting the second stroke moved the focus from {before} to {after}")
    await page.key("s")
    centred = await focus()
    if max(abs(a - b) for a, b in zip(centred, one[-1][1:])) > 1:
        failures.append(f"S left the focus at {centred}, not at the first stroke's last node, {one[-1][1:]}")
    await page.key("Delete")
    await count()
    if (selected := (await saved())[1]) != three[0][0]:
        failures.append(f"deleting the first stroke selected node {selected}, not the third's first, {three[0][0]}")

    await page.key("n")
    await drag(SECOND_TREE_ROW_PX)
    await count()
    await page.key("Delete")
    await count()
    if (emptied := await target()) != [SECOND_FIBER, 2, "empty class"]:
        failures.append(f"deleting fiber 2's only stroke left the target, trees and count {emptied}")
    before = await focus()
    await page.key("Delete")
    await count()
    selected, after, aimed = (await saved())[1], await focus(), await target()
    if [aimed[:2], selected] != [[FIBER, 1], three[-1][0]]:
        failures.append(f"Delete on the empty fiber 2 left the target and trees {aimed[:2]} and node {selected} selected, "
                        f"not fiber 1 alone and its last node, {three[-1][0]}")
    if after != before:
        failures.append(f"Delete on the empty fiber 2 moved the focus from {before} to {after}")
    await page.key("s")
    centred = await focus()
    if max(abs(a - b) for a, b in zip(centred, three[-1][1:])) > 1:
        failures.append(f"S left the focus at {centred}, not at fiber 1's last node, {three[-1][1:]}")
    await page.click("#points-undo")
    await count()
    if (undone := await target()) != [SECOND_FIBER, 2, "empty class"]:
        failures.append(f"undo left the target, trees and count {undone}, not the empty fiber 2 back")

    await page.key("O", shift=True)
    await click(*CLICK_PX)
    base = await count()
    await drag(FIRST_ROW_PX)
    first = await count() - base
    await drag(SECOND_ROW_PX)
    total = await count()
    second = total - base - first

    await click(0, FIRST_ROW_PX)
    if (got := await stroke()) != f"{first} nodes":
        failures.append(f"the first stroke selected reads {got!r}, not its {first} nodes")

    await page.key("Delete")
    deleted = [total - await count(), await stroke()]
    if deleted != [first, "1 node"]:
        failures.append(f"Delete took out {deleted[0]} nodes, selecting a stroke of {deleted[1]!r}, not {first} and the click")
    await page.click("#points-undo")
    restored = [await count(), await stroke()]
    if restored != [total, f"{first} nodes"]:
        failures.append(f"one undo left {restored}, not {total} nodes and the stroke of {first} selected")

    await click(0, SECOND_ROW_PX)
    await page.key("Delete", shift=True)
    if (left := [total - await count(), await stroke()]) != [1, f"{second - 1} nodes"]:
        failures.append(f"Shift+Delete took out {left[0]} nodes, leaving a stroke of {left[1]!r}, not 1 of {second}")
    await page.key("Backspace")
    if (gone := total - await count()) != second:
        failures.append(f"Backspace left {gone} of the second stroke's {second} nodes taken out, not all")

    await page.click("#points-clear")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.click("#points-tool")
    for index, row in enumerate(SCRIBBLE_ROWS_PX):
        if index in (0, 3):
            await page.key("n")
        await drag(row)
        await count()
    scribbles, _, classes = await saved()
    if len(scribbles) != 4 or classes != {FIBER_1: ids(*scribbles[:3]), FIBER_2: ids(scribbles[3])}:
        sizes = {name: len(nodes) for name, nodes in classes.items()}
        failures.append(f"after Clear all, N, three scribbles, N and one more, the classes hold {sizes} nodes, not three "
                        f"scribbles in {FIBER_1} and one in {FIBER_2}")
        return failures
    one, two, three, four = scribbles

    await click(-90, SCRIBBLE_ROWS_PX[1])
    await page.key("'", shift=True)
    stepped = [(await saved())[1], await centred_on(three), await read(SCRIBBLE)]
    if stepped != [three[0][0], True, ["3 of 3", f"{len(three)} nodes"]]:
        failures.append(f"Shift+' from the second scribble gives [the node selected, the focus on the third's middle, "
                        f"Selected scribble] {stepped}, not [{three[0][0]}, True, ['3 of 3', '{len(three)} nodes']]")
    await page.key("'", shift=True)
    note = await read("document.querySelector('#points-note').textContent")
    if (held := [(await saved())[1], note]) != [three[0][0], LAST]:
        failures.append(f"Shift+' at the third scribble gives [the node selected, the note] {held}, not "
                        f"[{three[0][0]}, {LAST!r}]")
    for _ in range(2):
        await page.key(";", shift=True)
    if (back := [(await saved())[1], await centred_on(one)]) != [one[0][0], True]:
        failures.append(f"Shift+; twice from the third scribble gives [the node selected, the focus on the first's "
                        f"middle] {back}, not [{one[0][0]}, True]")

    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.key("e")
    await click(0, OFF_PX)
    await page.key("Escape")
    await page.key("Tab")
    await page.key("'", shift=True)
    picker = await read("document.querySelector('#points-fiber').textContent")
    unselected = [picker, (await saved())[1], await centred_on(four), await read(SCRIBBLE)]
    if unselected != [SECOND_FIBER, four[0][0], True, ["1 of 1", f"{len(four)} nodes"]]:
        failures.append(f"Shift+' with no node selected gives [the picker's fiber, the node selected, the focus on the "
                        f"fourth's middle, Selected scribble] {unselected}, not [{SECOND_FIBER!r}, {four[0][0]}, True, "
                        f"['1 of 1', '{len(four)} nodes']]")

    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await click(-90, SCRIBBLE_ROWS_PX[0])
    await page.key("e")
    for moved, left in [([two], [one, three]), ([two, three], [one])]:
        await page.key("'", shift=True)
        picked = (await saved())[1]
        await page.key("x")
        await count()
        found = (await saved())[2]
        want = {FIBER_1: ids(*left), FIBER_2: ids(four), NO_INK: ids(*moved)}
        if [picked, found] != [moved[-1][0][0], want]:
            failures.append(f"in Edit Shift+' selects node {picked}, not scribble {len(moved) + 1}'s first, "
                            f"{moved[-1][0][0]}, and X leaves the scribbles in {scribbled(found, scribbles)}, not in "
                            f"{scribbled(want, scribbles)}")
            return failures
    return failures
