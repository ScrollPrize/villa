import asyncio
import json
import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, POLL_S, VOLUME, ct_key

ZOOM = 4
STEP_PX = 10
ROWS_PX = [-60, -20, 20, 60]
OFF_PX = 100
AWAY_XYZ = "20610.63, 14099.20, 43903.17"
DRAG_PX = 30
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser"
NOTE = "document.querySelector('#points-note').textContent"
STATUS = "document.querySelector('#status').textContent"
STATUS_S = 2.0
MODE = "document.querySelector('#points-mode [aria-pressed=true]')?.textContent ?? null"
NAME = "document.querySelector('#points-name').value"
LABEL = "document.querySelector('#tool-label').textContent"
FOCUS = "document.querySelector('#position-number').value"
PLACE = "document.querySelector('#points-scribble-place').textContent"
KOLLESIS = "document.querySelector('#points-mark-kollesis').getAttribute('aria-pressed')"
PICKER = """(() => {
  const at = (selector) => document.querySelector(selector);
  const role = at("#points-roles .active");
  return [at("#points-winding").textContent, at("#points-sides .active")?.textContent ?? null,
    at("#points-sheet").selectedOptions[0].textContent, role?.textContent ?? null];
})()"""
FIBER, INK, NO_INK = "S0_W0_HZ_RECTO_FIBER_1", "S0_W0_RECTO_SURFACE_INK", "S0_W0_RECTO_SURFACE_NOINK"
UP_NO_INK, UP_VERSO_NO_INK = "S0_W1_RECTO_SURFACE_NOINK", "S0_W1_VERSO_SURFACE_NOINK"
VERSO_NO_INK = "S0_W0_VERSO_SURFACE_NOINK"
FIBER_2 = "S0_W0_HZ_RECTO_FIBER_2"
SHOWN = {INK: ["W0", "recto", "S0", "Ink"], NO_INK: ["W0", "recto", "S0", "No ink"],
         UP_NO_INK: ["W+1", "recto", "S0", "No ink"], UP_VERSO_NO_INK: ["W+1", "verso", "S0", "No ink"],
         VERSO_NO_INK: ["W0", "verso", "S0", "No ink"]}
LABELS = {FIBER: "W0 › recto › fiber 1", INK: "W0 › recto › ink", NO_INK: "W0 › recto › no ink",
          UP_NO_INK: "W+1 › recto › no ink", UP_VERSO_NO_INK: "W+1 › verso › no ink",
          VERSO_NO_INK: "W0 › verso › no ink", FIBER_2: "W0 › recto › fiber 2"}
REFUSED = "This class holds 2 scribbles; C gives this scribble a class of its own, or Escape and Name renames them all."
STORED = f"""new Promise((resolve, reject) => {{
  const opening = indexedDB.open("teca-points");
  opening.onerror = () => reject(opening.error);
  opening.onsuccess = () => {{
    const read = opening.result.transaction("skeletons").objectStore("skeletons").get({json.dumps(VOLUME)});
    read.onerror = () => reject(read.error);
    read.onsuccess = () => {{ opening.result.close(); resolve(read.result ?? null); }};
  }};
}})"""


def classes(finest):
    return {thing.get("name"): {int(node.get("id")): (node.get("x"), node.get("y"), node.get("z"),
                                                      node.get("teca.strokeId")) for node in thing.iter("node")}
            for thing in ElementTree.fromstring(finest).findall("thing")}


def counts(found):
    return {name: len(nodes) for name, nodes in found.items()}


def holding(found, nodes):
    return sorted(name for name, held in found.items() if set(held) & set(nodes))


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

    async def shown():
        return [await page.evaluate(MODE), await page.evaluate(PICKER)]

    async def draw(row):
        await page.mouse("mouseMoved", x - 90, y + row)
        await page.mouse("mousePressed", x - 90, y + row, button="left", buttons=1, clickCount=1)
        for step_x in range(x - 90 + STEP_PX, x + 91, STEP_PX):
            await page.mouse("mouseMoved", step_x, y + row, button="left", buttons=1)
        await page.mouse("mouseReleased", x + 90, y + row, button="left", buttons=0, clickCount=1)
        await kept()

    async def click(row, dx=-90):
        await page.mouse("mouseMoved", x + dx, y + row)
        await page.mouse("mousePressed", x + dx, y + row, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", x + dx, y + row, button="left", buttons=0, clickCount=1)

    async def name(text, key):
        await page.type_into("points-name", text)
        await page.key(key)

    await page.click("#points-tool")
    for row in ROWS_PX[:3]:
        await draw(row)
    drawn = classes(await saved())
    scribbles = {}
    for id, node in sorted(drawn.get(FIBER, {}).items()):
        scribbles.setdefault(node[3], {})[id] = node
    if sorted(drawn) != [FIBER] or len(scribbles) != 3:
        failures.append(f"three scribbles drawn at load are {len(scribbles)} in {sorted(drawn)}, not three in {FIBER}")
        return failures
    first, second, third = scribbles.values()

    await page.key("e")
    await click(ROWS_PX[2])
    await page.key("c")
    status = await said(want := f"Scribble of {len(third)} nodes is Class 2.")
    await kept()
    found = classes(await saved())
    picked = [await page.evaluate(MODE), await page.evaluate(NAME), await page.evaluate(LABEL)]
    if status != want or found != {FIBER: {**first, **second}, "Class 2": third}:
        failures.append(f"C with the third scribble picked says {status!r} and leaves classes {counts(found)}, the "
                        f"third in {holding(found, third)}, not alone in Class 2, {out / SAVED}")
    if picked != ["Edit", "Class 2", "Edit: Class 2"]:
        failures.append(f"after C the mode, Name and the label beside the pointer are {picked}, not Class 2 picked")
    await name("lone", "Enter")
    await kept()
    if (found := classes(await saved())) != {FIBER: {**first, **second}, "lone": third}:
        failures.append(f"Name lone after C leaves classes {counts(found)}, not {FIBER} with the first two scribbles "
                        f"and lone with the third alone, {out / SAVED}")

    await click(ROWS_PX[1])
    before = await saved()
    await name("renamed", "Enter")
    refused = [await page.evaluate(NOTE), await page.evaluate(NAME)]
    if refused != [REFUSED, "fiber 1"] or await saved() != before:
        failures.append(f"Name with the second scribble picked, its class holding two, gives [the note, Name] "
                        f"{refused} and the classes {counts(classes(await saved()))}, not refused")
    before = await saved()
    await name("typed", "Escape")
    escaped = [await page.evaluate(NAME), await page.evaluate(MODE), await page.evaluate(LABEL)]
    if escaped != ["fiber 1", "Edit", f"Edit: {LABELS[FIBER]}"] or await saved() != before:
        failures.append(f"Escape in Name after typing gives [Name, the mode, the label] {escaped} and the classes "
                        f"{counts(classes(await saved()))}, not the class's name kept, the second scribble picked")

    await page.key("K", shift=True)
    marked = await page.evaluate(KOLLESIS)
    await page.key("Escape")
    released = [marked, await page.evaluate(KOLLESIS), await page.evaluate(MODE)]
    before = await saved()
    await page.key("x")
    if released != ["true", "false", "Draw"] or await saved() != before:
        failures.append(f"Shift+K then Escape with the second scribble picked give [Mark pressed, after Escape, the "
                        f"mode] {released}, and X then leaves classes {counts(classes(await saved()))}, which a "
                        f"scribble still picked changes")

    await page.key("e")
    await click(ROWS_PX[0])
    await page.key("'")
    await page.key("x")
    status = await said(want := f"Scribble of {len(first)} nodes is {LABELS[NO_INK]}.")
    await kept()
    found = classes(await saved())
    if status != want or found.get(NO_INK) != first:
        failures.append(f"' within the first scribble picked, then X, says {status!r} and leaves its nodes in "
                        f"{holding(found, first)}, not all in {NO_INK}")
    await click(0, dx=0)
    await page.key("a")
    status = await said(want := f"Scribble of {len(first)} nodes is {LABELS[INK]}.")
    await kept()
    before = await saved()
    await page.key("a")
    if (note := await page.evaluate(NOTE)) != f"This scribble is {LABELS[INK]} already." or await saved() != before:
        failures.append(f"A on the first scribble picked says {status!r}, and A again {note!r}, not that it is ink")

    await page.click("#points-clear")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.click("#points-tool")
    await page.key("a")
    await draw(ROWS_PX[0])
    await page.key("x")
    await draw(ROWS_PX[1])
    drawn = classes(await saved())
    first, second = drawn.get(INK, {}), drawn.get(NO_INK, {})
    if sorted(drawn) != [INK, NO_INK] or not first or not second or len({*first.values(), *second.values()}) < 2:
        failures.append(f"after Clear all, A, a scribble, X and another, the classes are {sorted(drawn)}, not {INK} "
                        f"and {NO_INK}")
        return failures

    await click(ROWS_PX[0])
    clicked = await shown()
    before = await saved()
    await page.key("]")
    stepped = await shown()
    if clicked != ["Draw", SHOWN[INK]] or stepped != ["Draw", ["W+1", "recto", "S0", "Ink"]] or \
            await saved() != before:
        failures.append(f"in Draw a click on the first scribble and ] show {clicked} and {stepped} and leave classes "
                        f"{counts(classes(await saved()))}, not its class and then the next scribble's W+1 ink")
    await page.key("e")
    if (whose := await shown()) != ["Edit", SHOWN[INK]]:
        failures.append(f"E after a click on the first scribble shows {whose}, not Edit and its class {SHOWN[INK]}")
    befores = []
    after = await saved()
    for key, into in [("x", NO_INK), ("]", UP_NO_INK), ("t", UP_VERSO_NO_INK)]:
        befores.append(after)
        await page.key(key)
        status = await said(want := f"Scribble of {len(first)} nodes is {LABELS[into]}.")
        await kept()
        after = await saved()
        found = classes(after)
        if status != want or found.get(into, {}) != {**first, **(second if into == NO_INK else {})}:
            failures.append(f"{key.upper()} says {status!r} and leaves the first scribble's nodes in "
                            f"{holding(found, first)}, not all in {into} as they were, {out / SAVED}")
            return failures
        if (whose := await shown()) != ["Edit", SHOWN[into]]:
            failures.append(f"after {key.upper()} the picker shows {whose}, not Edit and {SHOWN[into]}")
        if key == "x" and (stored := await page.evaluate(STORED)) != after.decode():
            failures.append(f"after X the browser keeps another NML than Save writes, {len(stored or '')} characters "
                            f"against {len(after)}")
    for before, back in zip(befores[:0:-1], [UP_NO_INK, NO_INK]):
        await page.click("#points-undo")
        await kept()
        same, whose = await saved() == before, await shown()
        if not same or whose != ["Edit", SHOWN[back]]:
            failures.append(f"after an undo the NML is {'' if same else 'not '}the one before the change undone, "
                            f"and the picker shows {whose}, not Edit and {SHOWN[back]}, {out / SAVED}")
    undone = classes(befores[1])

    await click(OFF_PX, dx=0)
    off = [await page.evaluate(LABEL), classes(await saved()) == undone]
    await page.mouse("mouseMoved", x - 90, y + ROWS_PX[1])
    await page.mouse("mousePressed", x - 90, y + ROWS_PX[1], button="left", buttons=1, clickCount=1)
    for step_px in range(STEP_PX, DRAG_PX + 1, STEP_PX):
        await page.mouse("mouseMoved", x - 90 + step_px, y + ROWS_PX[1] + step_px, button="left", buttons=1)
    await page.mouse("mouseReleased", x - 90 + DRAG_PX, y + ROWS_PX[1] + DRAG_PX, button="left", buttons=0,
                     clickCount=1)
    dragged = [await shown(), classes(await saved()) == undone]
    if off != ["Edit: click a scribble", True] or dragged != [["Edit", SHOWN[NO_INK]], True]:
        failures.append(f"in Edit a click off the scribbles gives [the label, the NML kept] {off}, and a drag of the "
                        f"second scribble's node [what shows, the NML kept] {dragged}, not nothing picked and then it")

    await page.key("Escape")
    escaped = await page.evaluate(MODE)
    await page.key("a")
    if escaped != "Draw" or await shown() != ["Draw", SHOWN[INK]] or classes(await saved()) != undone:
        failures.append(f"Escape leaves {escaped} and A shows {await shown()}, not Draw and the next scribble's "
                        f"{SHOWN[INK]}, or changes a scribble")
    await draw(ROWS_PX[2])
    found = classes(await saved())
    third = {id: node for id, node in found.get(INK, {}).items() if id not in first}
    if not third or holding(found, first) != [NO_INK]:
        failures.append(f"after Escape and A the scribble drawn is in {sorted(found)} and the first in "
                        f"{holding(found, first)}, not {INK} and {NO_INK}")

    await page.key("e")
    await click(ROWS_PX[1])
    await page.key("t")
    status = await said(want := f"Scribble of {len(second)} nodes is {LABELS[VERSO_NO_INK]}.")
    await kept()
    await page.key("Escape")
    await draw(ROWS_PX[3])
    before_a = await saved()
    found = classes(before_a)
    fourth = set(found.get(VERSO_NO_INK, {})) - set(second)
    whose = await shown()
    if status != want or whose != ["Draw", SHOWN[VERSO_NO_INK]] or not fourth or \
            holding(found, second) != [VERSO_NO_INK]:
        failures.append(f"E, a click on the second scribble and T say {status!r}, and after Escape and a scribble "
                        f"drawn the picker shows {whose}, and the two are in {holding(found, second)} and "
                        f"{holding(found, fourth)}, not {VERSO_NO_INK}")
    await page.key("a")
    last = await saved()
    if (whose := await shown()) != ["Draw", ["W0", "verso", "S0", "Ink"]] or last != before_a:
        failures.append(f"A after drawing shows {whose}, not the next scribble's verso ink, or changes a scribble")

    await page.click("#points-clear")
    opened_zip = page.browser.downloads / "opened.zip"
    opened_zip.write_bytes((out / SAVED).read_bytes())
    await page.choose_file("#points-file", opened_zip)
    await kept()
    if await saved() != last:
        failures.append(f"the zip opened saves another NML than it holds, {out / SAVED}")

    async def pan():
        await page.type_into("position-number", AWAY_XYZ)
        await page.key("Enter")

    async def selected():
        active = ElementTree.fromstring(await saved()).find("parameters/activeNode")
        return None if active is None else int(active.get("id"))

    async def centred(node):
        focus = [float(value) for value in (await page.evaluate(FOCUS)).split(",")]
        return max(abs(a - float(b)) for a, b in zip(focus, node[:3])) <= 1

    async def gone(scribble):
        ids = sorted(scribble)
        return [await selected(), await centred(scribble[ids[len(ids) // 2]]), await page.evaluate(PLACE)]

    async def listed(label):
        options = "[...document.querySelector('#points-class').options]"
        value = await page.evaluate(f"{options}.find((o) => o.text.startsWith({json.dumps(label + ' (')})).value")
        await page.choose("#points-class", value)

    await page.click("#points-clear")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.click("#points-tool")
    await page.click("#points-recto")
    await page.click("#points-fiber")
    for row in ROWS_PX[:2]:
        await draw(row)
    await page.key("n")
    await draw(ROWS_PX[2])
    drawn = classes(await saved())
    fibers = {FIBER: {}, FIBER_2: {}}
    for name, scribbles in fibers.items():
        for id, node in sorted(drawn.get(name, {}).items()):
            scribbles.setdefault(node[3], {})[id] = node
    if sorted(drawn) != [FIBER, FIBER_2] or [len(scribbles) for scribbles in fibers.values()] != [2, 1]:
        failures.append(f"after Clear all two scribbles in fiber 1, N and one are in {counts(drawn)}, not two in "
                        f"{FIBER} and one in {FIBER_2}")
        return failures
    (first, second), (third,) = (list(scribbles.values()) for scribbles in fibers.values())

    await pan()
    await listed(LABELS[FIBER])
    await page.key("s")
    if (got := await gone(second)) != (want := [min(second), True, "2 of 2"]):
        failures.append(f"with fiber 2's node selected the Class list's fiber 1 and S give [the node selected, the "
                        f"focus on fiber 1's second scribble's middle, its place] {got}, not {want}")
    await listed(LABELS[FIBER_2])
    await page.key("'", shift=True)
    if (got := await gone(third)) != (want := [min(third), True, "1 of 1"]):
        failures.append(f"with fiber 1's node selected the Class list's fiber 2 and Shift+' give [the node selected, "
                        f"the focus on fiber 2's scribble's middle, its place] {got}, not {want}")
    await page.key("Tab")
    await page.key("'", shift=True)
    if (got := await gone(first)) != (want := [min(first), True, "1 of 2"]):
        failures.append(f"with fiber 2's node selected Tab back to fiber 1 and Shift+' give [the node selected, the "
                        f"focus on fiber 1's first scribble's middle, its place] {got}, not {want}")

    start = min(first)
    await pan()
    await page.key("s")
    if (got := [await selected(), await centred(first[start])]) != [start, True]:
        failures.append(f"S with fiber 1's node {start} selected in fiber 1 gives [the node selected, the focus on it] "
                        f"{got}, not [{start}, True]")

    await page.key("n")
    before = await page.evaluate(FOCUS)
    for key, shift in [("s", False), ("'", True)]:
        await page.key(key, shift=shift)
        note = await page.evaluate(NOTE)
        stayed = [note, await page.evaluate(FOCUS), await selected()]
        if stayed != (want := ["W0 › recto › fiber 3 has no nodes yet.", before, start]):
            failures.append(f"{'Shift+' if shift else ''}{key.upper()} after N gives [the note, the focus, the node "
                            f"selected] {stayed}, not {want}")

    await pan()
    await page.key("e")
    await click(0, dx=0)
    await page.key("s")
    label = await page.evaluate(LABEL)
    if (got := [*await gone(second), label]) != (want := [min(second), True, "2 of 2", f"Edit: {LABELS[FIBER]}"]):
        failures.append(f"in Edit with no node selected S gives [the node selected, the focus on fiber 1's second "
                        f"scribble's middle, its place, the label] {got}, not {want}")
    return failures
