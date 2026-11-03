import asyncio
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, POLL_S, VOLUME, ct_key

ZOOM = 4
STEP_PX = 10
ROWS_PX = [-60, 0, 60]
CTRL = 2
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser"
NOTE = "document.querySelector('#points-note').textContent"
STATUS = "document.querySelector('#status').textContent"
STATUS_S = 2.0
LABEL = "document.querySelector('#tool-label').textContent"
RECTO, VERSO = "S0_W0_HZ_RECTO_FIBER_1", "S0_W0_VT_VERSO_FIBER_1"
VERSO_LABEL = "W0 › verso › fiber 1"


def classes(finest):
    return {thing.get("name"): sorted(int(node.get("id")) for node in thing.iter("node"))
            for thing in ElementTree.fromstring(finest).findall("thing")}


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
        return classes(finest)

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

    async def click(row, modifiers=0):
        await page.mouse("mouseMoved", x - 90, y + row, modifiers=modifiers)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1, modifiers=modifiers)
        await page.mouse("mouseReleased", x - 90, y + row, button="left", buttons=0, clickCount=1, modifiers=modifiers)

    await page.click("#points-tool")
    await page.click("#points-recto")
    await page.click("#points-fiber")
    nodes = []
    for row in ROWS_PX:
        before = sum((await saved()).values(), []) if nodes else []
        await draw(row)
        nodes.append(sorted(set(sum((await saved()).values(), [])) - set(before)))
    first, second, third = nodes

    await page.key("e")
    await click(ROWS_PX[0])
    for key, shift in [("Enter", False), ("'", True), ("'", True), ("Enter", False), (";", True)]:
        await page.key(key, shift=shift)
    await page.mouse("mouseMoved", x, y + ROWS_PX[1])
    label = await page.evaluate(LABEL)
    await page.key("t")
    status = await said(want := f"2 scribbles, {len(first + third)} nodes, are {VERSO_LABEL}.")
    await kept()
    found = await saved()
    if [label, status, found.get(VERSO), found.get(RECTO)] != \
            ["Edit 2 kept: W0 › recto › fiber 1", want, sorted(first + third), second]:
        failures.append(f"a click on the first scribble, Enter, two steps, Enter, a step back and T give the label "
                        f"{label!r}, say {status!r} and leave {found}, not the first and third in {VERSO} and the "
                        f"second in {RECTO}, {out / SAVED}")
        return failures

    await click(ROWS_PX[0], CTRL)
    await page.key("t")
    status = await said(want := f"Scribble of {len(third)} nodes is W0 › recto › fiber 1.")
    await kept()
    found = await saved()
    if [status, found.get(VERSO), found.get(RECTO)] != [want, first, sorted(second + third)]:
        failures.append(f"a Ctrl+click on the first scribble and T say {status!r} and leave {found}, not the third alone "
                        f"back in {RECTO}, {out / SAVED}")

    labels = []
    for key in [",", "."]:
        await page.key(key, shift=True)
        await page.mouse("mouseMoved", x, y + ROWS_PX[1])
        labels.append(await page.evaluate(LABEL))
    await page.key(".", shift=True)
    labels.append(await page.evaluate(NOTE))
    wanted = ["Edit 1 kept: W0 › recto › fiber 1", f"Edit 1 kept: {VERSO_LABEL}",
              "No class with nodes lies after this one in the Class list."]
    if labels != wanted:
        failures.append(f"Shift+, then Shift+. twice give {labels}, not {wanted}")
    return failures
