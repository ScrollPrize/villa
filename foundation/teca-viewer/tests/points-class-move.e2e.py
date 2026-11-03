import asyncio
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, POLL_S, VOLUME, ct_key

ZOOM = 4
STEP_PX = 10
ROWS_PX = [-80, -40, 0, 40, 80]
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser"
NOTE = "document.querySelector('#points-note').textContent"
STATUS = "document.querySelector('#status').textContent"
STATUS_S = 2.0
LABEL = "document.querySelector('#tool-label').textContent"
LISTED = "[...document.querySelector('#points-class').options].map((option) => option.text)"
FIBER_1, FIBER_2, FIBER_7 = "S0_W0_HZ_RECTO_FIBER_1", "S0_W0_HZ_RECTO_FIBER_2", "S0_W0_HZ_RECTO_FIBER_7"
VERSO_1, VERSO_2, UP_FIBER_2 = "S0_W0_VT_VERSO_FIBER_1", "S0_W0_VT_VERSO_FIBER_2", "S0_W1_HZ_RECTO_FIBER_2"
LABELS = {FIBER_1: "W0 › recto › fiber 1", FIBER_2: "W0 › recto › fiber 2", FIBER_7: "W0 › recto › fiber 7",
          VERSO_1: "W0 › verso › fiber 1", VERSO_2: "W0 › verso › fiber 2", UP_FIBER_2: "W+1 › recto › fiber 2"}


def classes(finest):
    return {thing.get("name"): (int(thing.get("id")),
                                {int(node.get("id")): node.get("teca.strokeId") for node in thing.iter("node")})
            for thing in ElementTree.fromstring(finest).findall("thing")}


def nodes(found, name):
    return sorted(found.get(name, (None, {}))[1])


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

    async def kept():
        return await page.wait_text(NOTE, lambda note: KEPT in note, 60)

    async def saved():
        path = await page.download("#points-save", SAVED)
        (out / SAVED).write_bytes(path.read_bytes())
        with zipfile.ZipFile(path) as zipped:
            finest = zipped.read(FINEST)
        path.unlink()
        return finest

    async def said(want):
        deadline = time.monotonic() + STATUS_S
        while (status := await page.evaluate(STATUS)) != want and time.monotonic() < deadline:
            await asyncio.sleep(POLL_S)
        return status

    async def draw(row):
        await page.mouse("mouseMoved", x - 90, y + row)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1)
        for step_x in range(x - 90 + STEP_PX, x + 91, STEP_PX):
            await page.mouse("mouseMoved", step_x, y + row, button="left", buttons=1)
        await page.mouse("mouseReleased", x + 90, y + row, button="left", buttons=0, clickCount=1)
        await kept()

    async def click(row):
        await page.mouse("mouseMoved", x - 90, y + row)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", x - 90, y + row, button="left", buttons=0, clickCount=1)

    async def whole(row):
        await click(row)
        await page.key("e", shift=True)
        return await page.evaluate(LABEL)

    async def undone(before, what):
        await page.click("#points-undo")
        await kept()
        if await saved() != before:
            failures.append(f"undo after {what} gives back another NML than the one before it, {out / SAVED}")

    await page.click("#points-tool")
    await page.click("#points-recto")
    await page.click("#points-fiber")
    for key, rows in [(None, ROWS_PX[:2]), ("n", ROWS_PX[2:4]), ("t", ROWS_PX[4:])]:
        if key is not None:
            await page.key(key)
        for row in rows:
            await draw(row)
    drawn = classes(await saved())
    strokes = {name: len(set(found.values())) for name, (_, found) in drawn.items()}
    if strokes != {FIBER_1: 2, FIBER_2: 2, VERSO_1: 1}:
        failures.append(f"two scribbles in fiber 1, N, two, T and one are scribbles {strokes} by class, not two in "
                        f"{FIBER_1} and {FIBER_2} and one in {VERSO_1}")
        return failures
    fiber_1, fiber_2, verso_1 = (nodes(drawn, name) for name in [FIBER_1, FIBER_2, VERSO_1])
    first = drawn[FIBER_2][1][fiber_2[0]]
    scribble = sorted(id for id, stroke in drawn[FIBER_2][1].items() if stroke == first)

    await page.key("e")
    label = await whole(ROWS_PX[2])
    before = await saved()
    await page.key("t")
    status = await said(want := f"Class of 2 scribbles, {len(fiber_2)} nodes, is {LABELS[VERSO_2]}.")
    await kept()
    found, listed = classes(await saved()), await page.evaluate(LISTED)
    gone = [text for text in listed if text.startswith(f"{LABELS[FIBER_2]} (")]
    if label != f"Edit class: {LABELS[FIBER_2]}":
        failures.append(f"a click on fiber 2 and Shift+E give the label {label!r}, not its whole class picked")
    if [status, nodes(found, VERSO_2), found.get(VERSO_2, (None,))[0], FIBER_2 in found, gone] != \
            [want, fiber_2, drawn[FIBER_2][0], False, []]:
        failures.append(f"T with recto fiber 2 picked whole says {status!r}, leaves {len(nodes(found, VERSO_2))} of "
                        f"its {len(fiber_2)} nodes in {VERSO_2}, its tree {drawn[FIBER_2][0]} as tree "
                        f"{found.get(VERSO_2, (None,))[0]}, and the Class list listing {gone}, not every node in "
                        f"{VERSO_2} as that tree with no {LABELS[FIBER_2]}, {out / SAVED}")
    await undone(before, "T with recto fiber 2 picked whole")

    label = await whole(ROWS_PX[0])
    before = await saved()
    await page.key("t")
    want = f"Class of 2 scribbles, {len(fiber_1)} nodes, is {LABELS[VERSO_2]} (fiber 1 is taken there)."
    status = await said(want)
    await kept()
    found = classes(await saved())
    if [label, status, nodes(found, VERSO_2), nodes(found, VERSO_1)] != \
            [f"Edit class: {LABELS[FIBER_1]}", want, fiber_1, verso_1]:
        failures.append(f"a click on fiber 1, Shift+E and T give the label {label!r} and say {status!r}, leaving "
                        f"{len(nodes(found, VERSO_2))} nodes in {VERSO_2} and {len(nodes(found, VERSO_1))} in "
                        f"{VERSO_1}, not fiber 1's {len(fiber_1)} apart from verso's own fiber 1, {out / SAVED}")
    await page.choose("#points-fibers", "1")
    status = await said(want := f"Class of 2 scribbles, {len(fiber_1)} nodes, joins {LABELS[VERSO_1]}.")
    await kept()
    found, listed = classes(await saved()), await page.evaluate(LISTED)
    gone = [text for text in listed if text.startswith(f"{LABELS[VERSO_2]} (")]
    if [status, nodes(found, VERSO_1), VERSO_2 in found, gone] != [want, sorted(fiber_1 + verso_1), False, []]:
        failures.append(f"fiber 1 in the Fiber list then says {status!r}, leaving {len(nodes(found, VERSO_1))} nodes "
                        f"in {VERSO_1} and the Class list listing {gone}, not all {len(fiber_1 + verso_1)} in one "
                        f"class, {out / SAVED}")
    await page.click("#points-undo")
    await kept()
    await undone(before, "T and the Fiber list's fiber 1 with recto fiber 1 picked whole")

    await whole(ROWS_PX[2])
    await page.key("]")
    status = await said(want := f"Class of 2 scribbles, {len(fiber_2)} nodes, is {LABELS[UP_FIBER_2]}.")
    await kept()
    found = classes(await saved())
    if [status, nodes(found, UP_FIBER_2), found.get(UP_FIBER_2, (None,))[0]] != [want, fiber_2, drawn[FIBER_2][0]]:
        failures.append(f"] with recto fiber 2 picked whole says {status!r} and leaves {len(nodes(found, UP_FIBER_2))} "
                        f"of its {len(fiber_2)} nodes in {UP_FIBER_2}, not all, its role and number kept, "
                        f"{out / SAVED}")
    await page.click("#points-undo")
    await kept()

    await page.mouse("mouseMoved", x - 90, y + ROWS_PX[2])
    await page.key("e", shift=True)
    label = await page.evaluate(LABEL)
    await page.key("t")
    status = await said(want := f"Scribble of {len(scribble)} nodes is {LABELS[VERSO_1]}.")
    await kept()
    found = classes(await saved())
    if [label, status, nodes(found, VERSO_1)] != [f"Edit: {LABELS[FIBER_2]}", want, sorted(verso_1 + scribble)]:
        failures.append(f"Shift+E with fiber 2 picked whole, then T, give the label {label!r} and say {status!r}, "
                        f"leaving {len(nodes(found, VERSO_1))} nodes in {VERSO_1}, not its first scribble's "
                        f"{len(scribble)} moved alone, {out / SAVED}")
    await page.click("#points-undo")
    await kept()

    await whole(ROWS_PX[0])
    await page.type_into("points-name", "fiber 7")
    await page.key("Enter")
    note = await page.evaluate(NOTE)
    if "in this browser" in note:
        await kept()
    named = await saved()
    found = classes(named)
    if [nodes(found, FIBER_7), found.get(FIBER_7, (None,))[0]] != [fiber_1, drawn[FIBER_1][0]]:
        failures.append(f"Name fiber 7 with recto fiber 1 picked whole says {note!r} and leaves "
                        f"{len(nodes(found, FIBER_7))} of its {len(fiber_1)} nodes in {FIBER_7}, not all, "
                        f"{out / SAVED}")
    await page.key("c")
    note = await page.evaluate(NOTE)
    if note != f"{LABELS[FIBER_7]} is a class of its own already." or await saved() != named:
        failures.append(f"C with fiber 7 picked whole says {note!r} and leaves classes "
                        f"{sorted(classes(await saved()))}, not that it is its own class, unchanged")
    return failures
