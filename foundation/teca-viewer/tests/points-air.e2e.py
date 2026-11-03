import asyncio
import base64
import json
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, NEUTRAL_XY, POLL_S, VOLUME, ct_key

ZOOM = 4
STEP_PX = 10
FIBER_ROW_PX, AIR_ROWS_PX, UP_AIR_ROW_PX = -80, [-40, 0], 40
CLICK_PX = (0, 80)
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser."
NOTE = "document.querySelector('#points-note').textContent"
STATUS = "document.querySelector('#status').textContent"
STATUS_S = 2.0
LABEL = "(() => { const e = document.querySelector('#tool-label'); return e.hidden ? null : e.textContent })()"
LABEL_MOVED = f"new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve({LABEL}))))"
LISTED = "[...document.querySelector('#points-class').options].map((option) => option.text)"
FIBER, AIR, UP_AIR = "W0 › recto › fiber 1", "W0–W+1 › air", "W+1–W+2 › air"
AIR_RGB = [13, 179, 255]
DIMMED_RGB = [6, 89, 128]
RGB_SLACK = 8
PIXELS = """(async ([png, points]) => {
  const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
    { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  return points.map(([x, y]) => [...context.getImageData(x, y, 1, 1).data.slice(0, 3)]);
})"""
NEXT_FIBER = "new fiber 99"
FIBER_BUTTON = """(() => { const e = document.querySelector('#points-fiber');
  return [e.textContent, e.clientWidth, e.scrollWidth] })()"""


def classes(finest):
    root = ElementTree.fromstring(finest)
    paths = {}

    def walk(element, path):
        for group in element.findall("group"):
            paths[group.get("id")] = [*path, group.get("name")]
            walk(group, paths[group.get("id")])

    walk(root.find("groups"), [])
    trees = {thing.get("name"): (paths.get(thing.get("groupId")),
                                 {int(node.get("id")): node.get("teca.strokeId") for node in thing.iter("node")})
             for thing in root.findall("thing")}
    return trees, {comment.get("node"): comment.get("content") for comment in root.iter("comment")}


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
    voxels_per_px = await page.config("BASE_VIEW_WIDTH_VOXELS") / ZOOM / min(rect["width"], rect["height"])

    async def kept():
        await page.wait_text(NOTE, lambda note: note.endswith(KEPT), 60)

    async def saved():
        path = await page.download("#points-save", SAVED)
        (out / SAVED).write_bytes(path.read_bytes())
        with zipfile.ZipFile(path) as zipped:
            finest = zipped.read(FINEST)
        path.unlink()
        return finest

    async def press(at_x, at_y):
        await page.mouse("mouseMoved", at_x, at_y)
        await page.mouse("mousePressed", at_x, at_y, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", at_x, at_y, button="left", buttons=0, clickCount=1)

    async def draw(row):
        await page.mouse("mouseMoved", x - 90, y + row)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1)
        for step_x in range(x - 90 + STEP_PX, x + 91, STEP_PX):
            await page.mouse("mouseMoved", step_x, y + row, button="left", buttons=1)
        await page.mouse("mouseReleased", x + 90, y + row, button="left", buttons=0, clickCount=1)
        await kept()

    async def labelled(key, shift=False):
        await page.mouse("mouseMoved", x - 90, y + UP_AIR_ROW_PX + 20)
        await page.key(key, shift=shift)
        return await page.evaluate(LABEL)

    async def focus():
        return [float(value) for value in (await page.evaluate("document.querySelector('#position-number').value")).split(",")]

    await page.click("#points-tool")
    marks = await page.evaluate("[...document.querySelectorAll('[id^=\"points-mark-\"]')].map((button) => button.textContent)")
    if marks != ["Kollesis"]:
        failures.append(f"the Mark row holds {marks}, not Kollesis alone")
    await page.mouse("mouseMoved", x + CLICK_PX[0], y + CLICK_PX[1])
    await page.key("B", shift=True)
    await press(x + CLICK_PX[0], y + CLICK_PX[1])
    await kept()
    await page.key("Escape")
    if (comments := classes(await saved())[1]) != {}:
        failures.append(f"Shift+B and a click leave the comments {comments}, not none")

    await draw(FIBER_ROW_PX)
    labels = [await labelled("a", shift=True), await labelled("Tab"), await labelled("Tab")]
    if labels != [AIR, FIBER, AIR]:
        failures.append(f"Shift+A, Tab and Tab give the labels {labels}, not {[AIR, FIBER, AIR]}")
        return failures
    for row in AIR_ROWS_PX:
        await draw(row)
    if (label := await labelled("]")) != UP_AIR:
        failures.append(f"] from {AIR} gives the label {label!r}, not {UP_AIR}")
        return failures
    await draw(UP_AIR_ROW_PX)
    listed = [text.rsplit(" (", 1)[0] for text in await page.evaluate(LISTED)]
    if listed != [FIBER, AIR, UP_AIR]:
        failures.append(f"the Class list lists {listed}, not {[FIBER, AIR, UP_AIR]}")

    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    png, _ = await page.picture(rect)
    (out / "air.png").write_bytes(png)
    firsts = [[x - 90 - round(rect["x"]), y + row - round(rect["y"])] for row in [UP_AIR_ROW_PX, AIR_ROWS_PX[0]]]
    bright, dimmed = await page.evaluate(f"{PIXELS}({json.dumps([base64.b64encode(png).decode(), firsts])})")
    if any(abs(got - want) > RGB_SLACK for got, want in zip(bright + dimmed, AIR_RGB + DIMMED_RGB)):
        failures.append(f"the first nodes of {UP_AIR} and {AIR} are {bright} and {dimmed}, not air's {AIR_RGB} and "
                        f"{DIMMED_RGB}, {out / 'air.png'}")

    drawn = classes(await saved())[0]
    paths = {name: path for name, (path, _) in drawn.items()}
    want = {"S0_W0_HZ_RECTO_FIBER_1": ["W0", "recto"], "W0_W1_AIR": ["W0–W+1"], "W1_W2_AIR": ["W+1–W+2"]}
    if paths != want:
        failures.append(f"the NML's trees and their groups are {paths}, not {want}, {out / SAVED}")
        return failures
    strokes = drawn["W0_W1_AIR"][1]
    last = sorted(id for id, stroke in strokes.items() if stroke == strokes[max(strokes)])
    up_air = sorted(drawn["W1_W2_AIR"][1])

    await page.click("#points-winding-down")
    before = await focus()
    label = await labelled("s")
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    after = await focus()
    stepped = await saved()
    found = classes(stepped)[0]
    selected = ElementTree.fromstring(stepped).find("parameters/activeNode").get("id")
    if [label, selected] != [AIR, str(last[0])]:
        failures.append(f"the picker's − and S give the label {label!r} and select node {selected}, not {AIR} and the "
                        f"first of its last scribble, {last[0]}")
    dx, dy = (before[0] - after[0]) / voxels_per_px, (after[1] - before[1]) / voxels_per_px
    await page.key("e")
    await press(round(x - 90 + dx), round(y + UP_AIR_ROW_PX + dy))
    await page.mouse("mouseMoved", round(x - 90 + dx) + 1, round(y + UP_AIR_ROW_PX + dy))
    picked = await page.evaluate(LABEL_MOVED)
    before_move = await saved()
    await page.browser.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "[", "code": "BracketLeft",
                                                       "windowsVirtualKeyCode": 219, "text": "["})
    await page.browser.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "[", "code": "BracketLeft",
                                                       "windowsVirtualKeyCode": 219})
    deadline = time.monotonic() + STATUS_S
    want_status = f"Scribble of {len(up_air)} nodes is {AIR}."
    while (status := await page.evaluate(STATUS)) != want_status and time.monotonic() < deadline:
        await asyncio.sleep(POLL_S)
    await kept()
    moved = classes(await saved())[0]
    if [picked, status, sorted(moved["W0_W1_AIR"][1]), sorted(moved["W1_W2_AIR"][1])] != \
            [f"Edit: {UP_AIR}", want_status, sorted([*found["W0_W1_AIR"][1], *up_air]), []]:
        failures.append(f"in Edit a click on {UP_AIR}'s scribble and [ give the label {picked!r}, say {status!r} and "
                        f"leave {len(moved['W0_W1_AIR'][1])} nodes in W0_W1_AIR and {len(moved['W1_W2_AIR'][1])} in "
                        f"W1_W2_AIR, not its {len(up_air)} nodes moved, {out / SAVED}")
    await page.click("#points-undo")
    await kept()
    finest = await saved()
    if finest != before_move:
        failures.append(f"undo after [ gives back another NML than the one before it, {out / SAVED}")

    listed = await page.evaluate(LISTED)
    await page.click("#points-clear")
    opened_zip = page.browser.downloads / "opened.zip"
    opened_zip.write_bytes((out / SAVED).read_bytes())
    await page.choose_file("#points-file", opened_zip)
    await kept()
    if (reopened := await page.evaluate(LISTED)) != listed:
        failures.append(f"the zip opened lists the classes {reopened}, not {listed}")
    if await saved() != finest:
        failures.append(f"the zip opened saves another NML than it holds, {out / SAVED}")

    await page.click("#points-draw")
    fiber = await page.evaluate("[...document.querySelector('#points-class').options]"
                                f".find((option) => option.text.startsWith({json.dumps(FIBER + ' (')})).value")
    await page.choose("#points-class", fiber)
    await page.type_into("points-name", "fiber 98")
    await page.key("Enter")
    await page.choose("#points-fibers", "99")
    text, width, needed = await page.evaluate(FIBER_BUTTON)
    if text != NEXT_FIBER or needed > width:
        failures.append(f"after Name's fiber 98 and the Fibers list's 99 the Fiber button shows {text!r} in {width} px "
                        f"where it needs {needed}, not {NEXT_FIBER} whole")
    return failures
