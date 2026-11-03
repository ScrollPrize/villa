import time
import xml.etree.ElementTree as ElementTree
import zipfile

from page import CENTER_XYZ, NEUTRAL_XY, VOLUME, ct_key, lines_key, surface_key

ZOOM = 4
COLUMNS_PX = [-135, -105, -75, -45, -15, 15, 45, 75, 105, 135]
ROWS_PX = [-60, 60]
# On the Surface, rows clear of its middle one, where the xy pane's plane cuts the segment and that pane's nodes may lie
# within the pen's reach: 150 px is about 3,800 voxels of z there.
SURFACE_COLUMNS_PX = [-120, -60, 0, 60, 120]
SURFACE_ROWS_PX = [-150, 150]
SAVED = "20260411134726-2.400um-0.2m-78keV-masked-points.zip"
FINEST = "20260411134726-2.400um-0.2m-78keV-masked.nml"
KEPT = ", kept in this browser."
START = ("W0", "recto", "S0", "new fiber 1")
PICKS = [
    ("#points-winding-up", None, ("W+1", "recto", "S0", "new fiber 1"), "W+1 › recto › fiber 1",
     "S0_W1_HZ_RECTO_FIBER_1", ["W+1", "recto"]),
    ("#points-verso", None, ("W+1", "verso", "S0", "new fiber 1"), "W+1 › verso › fiber 1",
     "S0_W1_VT_VERSO_FIBER_1", ["W+1", "verso"]),
    ("#points-sheet", "1", ("W+1", "verso", "S+1", "new fiber 1"), "W+1 › verso S+1 › fiber 1",
     "S1_W1_VT_VERSO_FIBER_1", ["W+1", "verso S+1"]),
    ("#points-ink", None, ("W+1", "verso", "S+1", "Ink"), "W+1 › verso S+1 › ink",
     "S1_W1_VERSO_SURFACE_INK", ["W+1", "verso S+1"]),
    ("#points-no-ink", None, ("W+1", "verso", "S+1", "No ink"), "W+1 › verso S+1 › no ink",
     "S1_W1_VERSO_SURFACE_NOINK", ["W+1", "verso S+1"]),
    ("#points-fiber", None, ("W+1", "verso", "S+1", "fiber 1"), "W+1 › verso S+1 › fiber 1",
     "S1_W1_VT_VERSO_FIBER_1", ["W+1", "verso S+1"]),
    ("#points-fibers", "2", ("W+1", "verso", "S+1", "new fiber 2"), "W+1 › verso S+1 › fiber 2",
     "S1_W1_VT_VERSO_FIBER_2", ["W+1", "verso S+1"]),
    ("#points-fibers", "1", ("W+1", "verso", "S+1", "fiber 1"), "W+1 › verso S+1 › fiber 1",
     "S1_W1_VT_VERSO_FIBER_1", ["W+1", "verso S+1"]),
    ("#points-ink", None, ("W+1", "verso", "S+1", "Ink"), "W+1 › verso S+1 › ink",
     "S1_W1_VERSO_SURFACE_INK", ["W+1", "verso S+1"]),
    ("#points-fiber", None, ("W+1", "verso", "S+1", "fiber 1"), "W+1 › verso S+1 › fiber 1",
     "S1_W1_VT_VERSO_FIBER_1", ["W+1", "verso S+1"]),
    ("#points-side-unknown", None, ("W+1", "?", "S+1", "new fiber 1"), "W+1 › side ? S+1 › fiber 1",
     "S1_W1_FIBER_1", ["W+1", "side ? S+1"]),
    ("#points-sheet", "?", ("W+1", "?", "S?", "new fiber 1"), "W+1 › side ? S? › fiber 1",
     "SN_W1_FIBER_1", ["W+1", "side ? S?"]),
    ("#points-winding-unknown", None, ("W?", "?", "S?", "new fiber 1"), "W? › side ? S? › fiber 1",
     "SN_WN_FIBER_1", ["W?", "side ? S?"]),
    ("#points-winding-unknown", None, ("W0", "?", "S?", "new fiber 1"), "W0 › side ? S? › fiber 1",
     "SN_W0_FIBER_1", ["W0", "side ? S?"]),
    ("#points-winding-down", None, ("W-1", "?", "S?", "new fiber 1"), "W-1 › side ? S? › fiber 1",
     "SN_W-1_FIBER_1", ["W-1", "side ? S?"]),
    ("#points-recto", None, ("W-1", "recto", "S?", "new fiber 1"), "W-1 › recto S? › fiber 1",
     "SN_W-1_HZ_RECTO_FIBER_1", ["W-1", "recto S?"]),
    ("#points-sheet", "0", ("W-1", "recto", "S0", "new fiber 1"), "W-1 › recto › fiber 1",
     "S0_W-1_HZ_RECTO_FIBER_1", ["W-1", "recto"]),
]
SURFACE_PICKS = [
    ("#points-winding-up", None, ("W+1", "recto", "S0", "Ink"), "W+1 › recto › ink",
     "S0_W1_RECTO_SURFACE_INK", ["W+1", "recto"]),
    ("#points-fibers", "1", ("W+1", "recto", "S0", "fiber 1"), "W+1 › recto › fiber 1",
     "S0_W1_HZ_RECTO_FIBER_1", ["W+1", "recto"]),
    ("#points-verso", None, ("W+1", "verso", "S0", "fiber 1"), "W+1 › verso › fiber 1",
     "S0_W1_VT_VERSO_FIBER_1", ["W+1", "verso"]),
    ("#points-sheet", "-1", ("W+1", "verso", "S-1", "new fiber 1"), "W+1 › verso S-1 › fiber 1",
     "S-1_W1_VT_VERSO_FIBER_1", ["W+1", "verso S-1"]),
    ("#points-no-ink", None, ("W+1", "verso", "S-1", "No ink"), "W+1 › verso S-1 › no ink",
     "S-1_W1_VERSO_SURFACE_NOINK", ["W+1", "verso S-1"]),
    ("#points-winding-down", None, ("W0", "verso", "S-1", "No ink"), "W0 › verso S-1 › no ink",
     "S-1_W0_VERSO_SURFACE_NOINK", ["W0", "verso S-1"]),
    ("#points-side-unknown", None, ("W0", "?", "S-1", "No ink"), "W0 › side ? S-1 › no ink",
     "S-1_W0_SURFACE_NOINK", ["W0", "side ? S-1"]),
    ("#points-winding-unknown", None, ("W?", "?", "S-1", "No ink"), "W? › side ? S-1 › no ink",
     "S-1_WN_SURFACE_NOINK", ["W?", "side ? S-1"]),
    ("#points-recto", None, ("W?", "recto", "S-1", "No ink"), "W? › recto S-1 › no ink",
     "S-1_WN_RECTO_SURFACE_NOINK", ["W?", "recto S-1"]),
    ("#points-ink", None, ("W?", "recto", "S-1", "Ink"), "W? › recto S-1 › ink",
     "S-1_WN_RECTO_SURFACE_INK", ["W?", "recto S-1"]),
]
KEYS = [
    ("]", ("W0", "recto", "S0", "new fiber 1")),
    ("a", ("W0", "recto", "S0", "Ink")),
    ("n", ("W0", "recto", "S0", "new fiber 1")),
    ("Tab", ("W0", "recto", "S0", "Ink")),
]
LISTS = [
    ("#points-sheet", [], None),
    ("#points-fibers", [], None),
    ("#points-joined", [], None),
    ("#points-class", [], None),
    ("#points-sheet", ["ArrowDown", "Enter"], ("W+4", "recto", "S0", "new fiber 1")),
]
PICKER = """(() => {
  const at = (selector) => document.querySelector(selector);
  const role = at("#points-roles .active");
  return [at("#points-winding").textContent, at("#points-sides .active")?.textContent ?? null,
    at("#points-sheet").selectedOptions[0].textContent, role?.textContent ?? null];
})()"""
WATCH = """(() => { window.__picked = []; document.addEventListener("input", (event) => {
  if (event.target.matches("select")) window.__picked.push(event.target.id) }, true); return true })()"""
OPEN = "document.querySelector(%r).matches(':open')"


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
    spots = [(x + dx, y + dy) for dy in ROWS_PX for dx in COLUMNS_PX]
    await page.click("#points-tool")
    if (shown := tuple(await page.evaluate(PICKER))) != START:
        failures.append(f"the picker shows {shown} at first, not {START}")

    async def pick_and_place(picks, spots, placed):
        for count, ((selector, value, want, named, _, _), (spot_x, spot_y)) in enumerate(zip(picks, spots), placed + 1):
            if value is None:
                await page.click(selector)
            else:
                await page.choose(selector, value)
            if (shown := tuple(await page.evaluate(PICKER))) != want:
                failures.append(f"after {selector} {value or ''} the picker shows {shown}, not {want}")
            await page.mouse("mouseMoved", spot_x, spot_y)
            label = await page.evaluate("(() => { const e = document.querySelector('#tool-label'); "
                                        "return e.hidden ? null : e.textContent })()")
            if label != named:
                failures.append(f"after {selector} {value or ''} the label beside the pointer reads {label!r}, not {named!r}")
            await page.mouse("mousePressed", spot_x, spot_y, button="left", buttons=1, clickCount=1)
            await page.mouse("mouseReleased", spot_x, spot_y, button="left", buttons=0, clickCount=1)
            nodes = f", {count} node{'' if count == 1 else 's'}, "
            await page.wait_text("document.querySelector('#points-note').textContent",
                                 lambda note: nodes in note and note.endswith(KEPT), 60)

    await pick_and_place(PICKS, spots, 0)

    await page.mouse("mouseMoved", *NEUTRAL_XY)
    for key, want in KEYS:
        await page.key(key)
        if (shown := tuple(await page.evaluate(PICKER))) != want:
            failures.append(f"{key} left the picker showing {shown}, not {want}")

    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    area = (await page.panes())["surface"]
    x, y = round(area["x"] + area["width"] / 2), round(area["y"] + area["height"] / 2)
    await pick_and_place(SURFACE_PICKS, [(x + dx, y + dy) for dy in SURFACE_ROWS_PX for dx in SURFACE_COLUMNS_PX],
                         len(PICKS))

    await page.click("#points-winding-unknown")
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.key("n")
    await page.evaluate(WATCH)
    winding, sheet = 0, "S-1"
    for selector, choice, chosen in LISTS:
        named = f"{selector} closed by {' and '.join(choice) or 'Escape'}"
        await page.click(selector)
        if not await page.evaluate(OPEN % selector):
            failures.append(f"a click on {selector} did not open its list")
        for key in choice or ["Escape"]:
            await page.key(key)
        if await page.evaluate(OPEN % selector):
            failures.append(f"{named} is still open")
        if chosen is not None:
            sheet = chosen[2]
            if (shown := tuple(await page.evaluate(PICKER))) != chosen:
                failures.append(f"{named} left the picker showing {shown}, not {chosen}")
        kept = await page.evaluate(f"(window.__picked = [], document.querySelector({selector!r}).value)")
        for key, step, role in [("]", 1, "new fiber 1"), ("a", 0, "Ink"), ("n", 0, "new fiber 1")]:
            winding += step
            want = (f"W+{winding}", "recto", sheet, role)
            await page.key(key)
            if (shown := tuple(await page.evaluate(PICKER))) != want:
                failures.append(f"{key} after {named} left the picker showing {shown}, not {want}")
        value, picked = await page.evaluate(f"[document.querySelector({selector!r}).value, window.__picked]")
        if value != kept or picked:
            failures.append(f"], A and N after {named} changed its value from {kept!r} to {value!r}, picking in {picked}")

    path = await page.download("#points-save", SAVED)
    (out / SAVED).write_bytes(path.read_bytes())
    with zipfile.ZipFile(path) as zipped:
        root = ElementTree.fromstring(zipped.read(FINEST))
    paths = {}

    def walk(element, path):
        for group in element.findall("group"):
            paths[group.get("id")] = [*path, group.get("name")]
            walk(group, paths[group.get("id")])

    walk(root.find("groups"), [])
    trees = {int(node.get("id")): (thing.get("name"), paths.get(thing.get("groupId")))
             for thing in root.findall("thing") for node in thing.iter("node")}
    joined = [trees[node_id] for node_id in sorted(trees)]
    wanted = [(name, groups) for *_, name, groups in PICKS + SURFACE_PICKS]
    for index, ((selector, value, *_), got, want) in enumerate(zip(PICKS + SURFACE_PICKS, joined, wanted)):
        if got != want:
            pane = "xy" if index < len(PICKS) else "the Surface"
            failures.append(f"the node placed on {pane} after {selector} {value or ''} is in {got}, not {want}")
    if len(joined) != len(wanted):
        failures.append(f"the NML holds {len(joined)} nodes, not the {len(wanted)} placed, {out / SAVED}")
    return failures
