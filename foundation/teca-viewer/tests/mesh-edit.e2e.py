import asyncio
import json
import struct
import time
import zipfile
from array import array

import aiohttp
from page import CENTER_XYZ, NEUTRAL_XY, SURFACE, VOLUME, ct_key, lines_key, surface_key

ZOOM = 4
RADIUS = 128
DRAG_PX = (40, 40)
STEPS = 4
CENTRE_CELL = (812, 1040)
SAVED = "20231221180251-on-20260411134726-2.4um-edited.zip"
KEPT = ", kept in this browser."
BROUGHT = "Brought back the edits kept here before; Discard forgets them."
FORGOT = "Forgot the edits; the surface is drawn from its files."
RECORD_BYTES = 16384           # a drag of RADIUS moves about 130 vertices, 28 bytes each
GROW_RADIUS = 3 * RADIUS
GROWN_VOXELS = 8               # how far a vertex grown back may be from the files: 5.5 across the 260 voxels of the hole
SURFACE_WHEELS = 30
SURFACE_DRAG_PX = 20
NORMAL_COSINE = 0.99
ACROSS_VOXELS = 0.01           # a vertex pushed off the grabbed one's line: float32 rounding at x of 20000 is 0.002
RECORDS = """new Promise((resolve, reject) => {
  const request = indexedDB.open("teca-mesh-edit");
  request.onerror = () => reject(request.error);
  request.onsuccess = () => {
    const all = request.result.transaction("edits").objectStore("edits").getAll();
    all.onerror = () => reject(all.error);
    all.onsuccess = () => {
      request.result.close();
      resolve(all.result.map((edit) => edit.indices.byteLength + edit.before_N3.byteLength + edit.after_N3.byteLength));
    };
  };
})"""


def tiff_values(data):
    if data[:4] != b"II*\0":
        raise ValueError(f"not a little-endian TIFF: {data[:4]!r}")
    ifd = struct.unpack_from("<I", data, 4)[0]
    tags = {}
    for entry in range(struct.unpack_from("<H", data, ifd)[0]):
        tag, kind, _, value = struct.unpack_from("<HHII", data, ifd + 2 + 12 * entry)
        tags[tag] = value & 0xFFFF if kind == 3 else value
    height, width = tags[257], tags[256]
    values = array("f")
    values.frombytes(data[tags[273]:tags[273] + tags[279]])
    return values, (height, width)


def changed_cells(before, after, width):
    rows = [row for row in range(len(before["x"]) // width)
            if any(before[axis][row * width:(row + 1) * width] != after[axis][row * width:(row + 1) * width] for axis in "xyz")]
    return [cell for row in rows for cell in range(row * width, (row + 1) * width)
            if any(before[axis][cell] != after[axis][cell] for axis in "xyz")]


def surface_of(path):
    with zipfile.ZipFile(path) as saved:
        names = saved.namelist()
        directory = names[0].split("/")[0]
        meta = json.loads(saved.read(f"{directory}/meta.json"))
        axes = {axis: tiff_values(saved.read(f"{directory}/{axis}.tif")) for axis in "xyz"}
    grids = {grid for _, grid in axes.values()}
    if len(grids) != 1:
        raise ValueError(f"{path} has x, y and z of grids {grids}")
    return {axis: values for axis, (values, _) in axes.items()}, grids.pop(), meta


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    if (place := await page.place("#mesh-edit-section")) != "Mesh edit":
        failures.append(f"its card is on {place!r}, not a Mesh edit tab of its own")
    await page.reveal("#mesh-edit-section")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", str(ZOOM))
    await page.settle(t0, {"panes complete": ct_key})

    rect = (await page.panes())["xy"]
    x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
    voxels_per_px = await page.config("BASE_VIEW_WIDTH_VOXELS") / ZOOM / min(rect["width"], rect["height"])

    async def picture(name):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"panes complete": ct_key, "lines settled": lines_key})
        png, pixels = await page.picture(rect)
        (out / f"{name}.png").write_bytes(png)
        return pixels["sha256"]

    async def note_is(text):
        await page.wait_text("document.querySelector('#mesh-edit-note').textContent", lambda note: note == text, 120)

    async def kept():
        await page.wait_text("document.querySelector('#mesh-edit-note').textContent", lambda note: note.endswith(KEPT), 60)

    async def drag(by_px=DRAG_PX):
        await page.mouse("mouseMoved", x, y)
        await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
        for step in range(1, STEPS + 1):
            await page.mouse("mouseMoved", x + by_px[0] * step // STEPS, y + by_px[1] * step // STEPS,
                             button="left", buttons=1)
        await page.mouse("mouseReleased", x + by_px[0], y + by_px[1], button="left", buttons=0, clickCount=1)
        released = time.monotonic()
        await kept()
        kept_s = time.monotonic() - released
        settled, _ = await page.settle(released, {"lines settled": lines_key})
        return kept_s, settled["lines settled"]

    async def save(name):
        started = time.monotonic()
        path = await page.download("#mesh-edit-save", SAVED)
        print(f"{name} is saved {time.monotonic() - started:.2f} s after the click")
        (out / name).write_bytes(path.read_bytes())
        path.unlink()
        return surface_of(out / name)

    await page.add_surface()
    files = await picture("files")
    listed = await page.evaluate("[...document.querySelectorAll('#mesh-edit-surface option')].map((o) => o.value)")
    if listed != [SURFACE.rstrip("/")]:
        failures.append(f"Mesh edits lists {listed} for the surface added")
    await page.click("#mesh-edit-open")
    await page.wait_text("document.querySelector('#mesh-edit-note').textContent", lambda note: " vertices, " in note, 120)
    memory = await picture("memory")
    if memory != files:
        failures.append(f"the surface drawn from memory differs from its files, {out / 'memory.png'}")

    await page.click("#mesh-edit-move")
    kept_s, drawn_s = await drag()
    print(f"a drag of radius {RADIUS} voxels is kept {kept_s:.2f} s and its lines settle {drawn_s:.2f} s after the release")
    dragged = await picture("dragged")
    if dragged == memory:
        failures.append(f"the drag left the xy pane as it was, {out / 'dragged.png'}")
    moved, grid, meta = await save("dragged.zip")
    await page.click("#mesh-edit-undo")
    await kept()
    undone = await picture("undone")
    if undone != memory:
        failures.append(f"the undo drew another picture than before the drag, {out / 'undone.png'}")
    before, _, _ = await save("undone.zip")

    async def fetched(http, axis):
        async with http.get(f"{SURFACE.rstrip('/')}/{axis}.tif") as response:
            response.raise_for_status()
            return await response.read()

    async with aiohttp.ClientSession() as http:
        tiffs = await asyncio.gather(*(fetched(http, axis) for axis in "xyz"))
    for axis, data in zip("xyz", tiffs):
        values, files_grid = tiff_values(data)
        if files_grid != grid or values.tobytes() != before[axis].tobytes():
            failures.append(f"the save after the undo is not the files' {axis}.tif bit for bit, {out / 'undone.zip'}")
    width = grid[1]
    centre = CENTRE_CELL[0] * width + CENTRE_CELL[1]
    drag_XYZ = [DRAG_PX[0] * voxels_per_px, -DRAG_PX[1] * voxels_per_px, 0]
    grabbed = [moved[axis][centre] - before[axis][centre] for axis in "xyz"]
    if max(abs(a - b) for a, b in zip(grabbed, drag_XYZ)) > 0.01:
        failures.append(f"the vertex grabbed moved by {grabbed}, not the drag's {drag_XYZ}")
    scale_u, scale_v = meta["scale"]
    changed = changed_cells(before, moved, width)
    outside = [divmod(cell, width) for cell in changed
               if ((cell // width - CENTRE_CELL[0]) / scale_v) ** 2 + ((cell % width - CENTRE_CELL[1]) / scale_u) ** 2 >= RADIUS ** 2]
    if outside or len(changed) < 2:
        failures.append(f"the drag changed {len(changed)} vertices, {outside[:5]} of them outside its radius")

    await page.click("#mesh-edit-redo")
    await kept()
    if await picture("redone") != dragged:
        failures.append(f"the redo drew another picture than the drag, {out / 'redone.png'}")
    await drag()
    again = await picture("again")
    sizes = await page.evaluate(RECORDS)
    print(f"the browser keeps edits of {sizes} bytes")
    if len(sizes) != 2 or max(sizes) > RECORD_BYTES:
        failures.append(f"the browser keeps edits of {sizes} bytes, not two of at most {RECORD_BYTES}")
    await page.click("#mesh-edit-undo")
    await kept()
    await page.open(await page.address())
    await note_is(BROUGHT)
    if await picture("restored") != dragged:
        failures.append(f"the reload drew another picture than the drags not taken back, {out / 'restored.png'}")
    await page.click("#mesh-edit-redo")
    await kept()
    if await picture("restored-redone") != again:
        failures.append(f"the redo after the reload drew another picture than both drags, {out / 'restored-redone.png'}")

    whole, _, _ = await save("again.zip")
    await page.click("#mesh-edit-smooth")
    await drag()
    if await picture("smoothed") == again:
        failures.append(f"the smoothing left the xy pane as it was, {out / 'smoothed.png'}")
    await page.click("#mesh-edit-undo")
    await kept()

    await page.click("#mesh-edit-erase")
    await drag((0, 0))
    erased_picture = await picture("erased")
    if erased_picture == again:
        failures.append(f"the erase left the xy pane as it was, {out / 'erased.png'}")
    erased, _, _ = await save("erased.zip")
    hole = changed_cells(whole, erased, width)
    kept_on = [divmod(cell, width) for cell in hole if any(erased[axis][cell] != -1 for axis in "xyz")]
    rows_across = max(cell // width for cell in hole) - min(cell // width for cell in hole) if hole else 0
    columns_across = max(cell % width for cell in hole) - min(cell % width for cell in hole) if hole else 0
    print(f"the erase takes {len(hole)} vertices off the surface, {rows_across + 1} rows by {columns_across + 1} columns")
    if len(hole) < 2 or kept_on or rows_across > 2 * RADIUS * scale_v or columns_across > 2 * RADIUS * scale_u:
        failures.append(f"the erase changed {len(hole)} vertices across {rows_across} rows and {columns_across} columns, "
                        f"{kept_on[:5]} of them left on the surface")

    await page.choose("#mesh-edit-radius-number", str(GROW_RADIUS))
    await page.click("#mesh-edit-grow")
    await drag((0, 0))
    if await picture("grown") == erased_picture:
        failures.append(f"the growing left the xy pane as it was, {out / 'grown.png'}")
    grown, _, _ = await save("grown.zip")
    left_off = [divmod(cell, width) for cell in hole if all(grown[axis][cell] == -1 for axis in "xyz")]
    apart = max((sum((grown[axis][cell] - before[axis][cell]) ** 2 for axis in "xyz") ** 0.5 for cell in hole), default=0)
    print(f"growing puts the erased vertices back at most {apart:.2f} voxels from the files")
    if left_off or apart > GROWN_VOXELS:
        failures.append(f"growing left {left_off[:5]} of {len(hole)} erased vertices off the surface, and one {apart:.2f} "
                        f"voxels from the files")

    await page.choose("#mesh-edit-radius-number", str(RADIUS))
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    surface_rect = (await page.panes())["surface"]
    sx, sy = round(surface_rect["x"] + surface_rect["width"] / 2), round(surface_rect["y"] + surface_rect["height"] / 2)
    await page.mouse("mouseMoved", sx, sy)
    for _ in range(SURFACE_WHEELS):
        await page.mouse("mouseWheel", sx, sy, deltaX=0, deltaY=-240)

    async def surface_picture(name):
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        await page.settle(time.monotonic(), {"surface": surface_key})
        png, pixels = await page.picture(surface_rect)
        (out / f"{name}.png").write_bytes(png)
        return pixels["sha256"]

    await page.click("#mesh-edit-move")
    unpushed = await surface_picture("surface")
    await page.mouse("mouseMoved", sx, sy)
    if await page.evaluate("document.querySelector('.pane-surface .pane-reach').hidden"):
        failures.append("the Surface pane shows no ring about the pointer while the tool is set")
    await page.mouse("mousePressed", sx, sy, button="left", buttons=1, clickCount=1)
    for step in range(1, STEPS + 1):
        await page.mouse("mouseMoved", sx, sy - SURFACE_DRAG_PX * step // STEPS, button="left", buttons=1)
    await page.mouse("mouseReleased", sx, sy - SURFACE_DRAG_PX, button="left", buttons=0, clickCount=1)
    await kept()
    if await surface_picture("surface-pushed") == unpushed:
        failures.append(f"the push left the Surface pane as it was, {out / 'surface-pushed.png'}")
    pushed, _, _ = await save("surface-pushed.zip")
    shifts = {cell: [pushed[axis][cell] - grown[axis][cell] for axis in "xyz"] for cell in changed_cells(grown, pushed, width)}
    if len(shifts) < 2:
        failures.append(f"the push on the Surface pane moved {len(shifts)} vertices")
    else:
        top = max(shifts, key=lambda cell: sum(value * value for value in shifts[cell]))
        length = sum(value * value for value in shifts[top]) ** 0.5
        unit = [value / length for value in shifts[top]]
        u = [grown[axis][top + 1] - grown[axis][top - 1] for axis in "xyz"]
        v = [grown[axis][top + width] - grown[axis][top - width] for axis in "xyz"]
        normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        cosine = abs(sum(a * b for a, b in zip(normal, unit))) / sum(value * value for value in normal) ** 0.5
        along = {cell: sum(a * b for a, b in zip(shift, unit)) for cell, shift in shifts.items()}
        across = max(sum((shift[axis] - along[cell] * unit[axis]) ** 2 for axis in range(3)) ** 0.5
                     for cell, shift in shifts.items())
        outside = [divmod(cell, width) for cell in shifts
                   if ((cell // width - top // width) / scale_v) ** 2 + ((cell % width - top % width) / scale_u) ** 2
                   >= RADIUS ** 2]
        print(f"the push on the Surface pane moves {len(shifts)} vertices, the one grabbed {length:.2f} voxels at "
              f"{cosine:.4f} of the normal, the rest at most {across:.4f} voxels off its line")
        if cosine < NORMAL_COSINE or across > ACROSS_VOXELS or outside:
            failures.append(f"the push moved vertex {divmod(top, width)} at {cosine:.4f} of the normal, one {across:.4f} "
                            f"voxels off its line, and {outside[:5]} outside the radius")
    await page.click("#mesh-edit-undo")
    await kept()
    if await surface_picture("surface-undone") != unpushed:
        failures.append(f"the undo of the push drew another Surface than before it, {out / 'surface-undone.png'}")
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")

    await page.click("#mesh-edit-discard")
    await note_is(FORGOT)
    discarded = await picture("discarded")
    if len(page.browser.dialogs) != 1:
        failures.append(f"Discard asked {page.browser.dialogs}, not once")
    if discarded != files:
        failures.append(f"the discard did not draw the surface from its files, {out / 'discarded.png'}")
    return failures
