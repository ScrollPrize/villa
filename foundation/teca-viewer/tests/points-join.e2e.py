import asyncio
import json
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, POLL_S, VOLUME, ct_key

ZOOM = 4
STEP_PX = 10
STROKES_PX = [-60, 0, 60]
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser"
NOTE = "document.querySelector('#points-note').textContent"
STATUS = "document.querySelector('#status').textContent"
WINDING = "document.querySelector('#points-winding').textContent"
PRESSED = "document.querySelector('#points-join').getAttribute('aria-pressed')"
STATUS_S = 2.0
ARMED = "Join W+100: choose the winding it is, then Join again; Escape cancels."
MODE = "document.querySelector('#points-mode [aria-pressed=true]')?.textContent ?? null"
INKS = ["S0_W0_RECTO_SURFACE_INK", "S0_W1_RECTO_SURFACE_INK", "S0_W100_RECTO_SURFACE_INK"]
STORED = f"""new Promise((resolve, reject) => {{
  const opening = indexedDB.open("teca-points");
  opening.onerror = () => reject(opening.error);
  opening.onsuccess = () => {{
    const read = opening.result.transaction("skeletons").objectStore("skeletons").get({json.dumps(VOLUME)});
    read.onerror = () => reject(read.error);
    read.onsuccess = () => {{ opening.result.close(); resolve(read.result ?? null); }};
  }};
}})"""


def trees(finest):
    found = {}
    for thing in ElementTree.fromstring(finest).findall("thing"):
        nodes = {int(node.get("id")): (node.get("x"), node.get("y"), node.get("z"), node.get("teca.strokeId"))
                 for node in thing.iter("node")}
        edges = sorted(tuple(sorted((int(edge.get("source")), int(edge.get("target"))))) for edge in thing.iter("edge"))
        found[thing.get("name")] = (nodes, edges)
    return found


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

    async def click(row):
        await page.mouse("mouseMoved", x - 90, y + row)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", x - 90, y + row, button="left", buttons=0, clickCount=1)

    async def stroke(row):
        await page.mouse("mouseMoved", x - 90, y + row)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1)
        for step_x in range(x - 90 + STEP_PX, x + 91, STEP_PX):
            await page.mouse("mouseMoved", step_x, y + row, button="left", buttons=1)
        await page.mouse("mouseReleased", x + 90, y + row, button="left", buttons=0, clickCount=1)
        await kept()

    await page.click("#points-tool")
    await page.key("a")
    await stroke(STROKES_PX[0])
    await page.key("]")
    await stroke(STROKES_PX[1])
    await page.click("#points-new-region")
    if (winding := await page.evaluate(WINDING)) != "W+100":
        failures.append(f"New region made the target's winding {winding}, not W+100")
    await stroke(STROKES_PX[2])
    drawn = trees(await saved())
    if sorted(drawn) != sorted(INKS) or any(len(nodes) == 0 for nodes, _ in drawn.values()):
        failures.append(f"the three scribbles drawn are classes {sorted(drawn)}, not one in each of {INKS}")
        return failures
    await page.key("e")
    await click(STROKES_PX[2])
    shown = []
    for press in ["#points-join", "Escape", "#points-join", "#points-join", "#points-join", "click"]:
        if press == "click":
            await click(STROKES_PX[2])
        else:
            await (page.key(press) if press == "Escape" else page.click(press))
        shown.append([await page.evaluate(PRESSED), await page.evaluate(NOTE), await page.evaluate(MODE)])
    if [pressed for pressed, _, _ in shown] != ["true", "false", "true", "false", "true", "true"] or \
            [note == ARMED for _, note, _ in shown] != [True, False, True, False, True, True] or \
            any(mode != "Draw" for _, _, mode in shown):
        failures.append(f"after E, a click on the W+100 scribble, Join, Escape, Join, Join, Join and a click on it "
                        f"again give [pressed, the note, the mode] {shown}")
    await page.key("Tab")
    if (tabbed := [await page.evaluate(WINDING), await page.evaluate(PRESSED)]) != ["W+1", "true"]:
        failures.append(f"Tab while Join is pressed gives [the winding, pressed] {tabbed}, not ['W+1', 'true']")
    before = await saved()
    if trees(before) != drawn:
        counts = {name: len(nodes) for name, (nodes, _) in trees(before).items()}
        failures.append(f"before the second Join the classes hold {counts} nodes, not as drawn: a scribble moved")
        return failures

    await page.click("#points-join")
    deadline = time.monotonic() + STATUS_S
    while not (status := await page.evaluate(STATUS)).startswith("Joined") and time.monotonic() < deadline:
        await asyncio.sleep(POLL_S)
    if [status, await page.evaluate(PRESSED)] != ["Joined W+100 as W+1.", "false"]:
        failures.append(f"the second Join says {status!r} on the status bar and leaves Join pressed "
                        f"{await page.evaluate(PRESSED)}")
    await kept()
    after = await saved()
    joined = trees(after)
    want = {INKS[0]: drawn[INKS[0]],
            INKS[1]: ({**drawn[INKS[1]][0], **drawn[INKS[2]][0]}, sorted(drawn[INKS[1]][1] + drawn[INKS[2]][1]))}
    left = b'name="W+100"' in after
    if joined != want or left:
        counts = {name: len(nodes) for name, (nodes, _) in joined.items()}
        wanted = {name: len(nodes) for name, (nodes, _) in want.items()}
        failures.append(f"after the join the NML's trees hold {counts} nodes, not {wanted} as drawn, or it keeps a "
                        f"group of W+100: {left}")
    if (stored := await page.evaluate(STORED)) != after.decode():
        failures.append(f"after the join the browser keeps another NML than Save writes, {len(stored or '')} "
                        f"characters against {len(after)}")

    await page.click("#points-undo")
    await kept()
    if await saved() != before:
        failures.append(f"one undo after the join saves another NML than before it, {out / SAVED}")
    await page.click("#points-redo")
    await kept()
    if await saved() != after:
        failures.append(f"redo saves another NML than the join did, {out / SAVED}")

    await page.click("#points-clear")
    opened_zip = page.browser.downloads / "opened.zip"
    opened_zip.write_bytes((out / SAVED).read_bytes())
    await page.choose_file("#points-file", opened_zip)
    await kept()
    if await saved() != after:
        failures.append(f"the zip opened saves another NML than it holds, {out / SAVED}")
    return failures
