import asyncio
import json
import re
import time

from page import CENTER_XYZ, VOLUME, ct_key

ZOOM = 4
CLICKS_PX = [(-40, -20), (40, 20), (40, -20)]
KEPT = ", kept in this browser."
SECOND_S = 30
CLEARED_S = 1.0
OTHER = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/"
         "ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/")
NOTE = "document.querySelector('#points-note').textContent"
SECOND_DOCUMENT = "document.querySelector('#second')?.contentDocument"
SECOND_NOTE = f"({SECOND_DOCUMENT}?.querySelector('#points-note')?.textContent ?? '')"
WATCHED = "Another tab has this CT's nodes open; the 2 nodes kept show here"
BROUGHT = "Brought back 3 nodes placed here before."
EDITS = ["#points-tool", "#points-new-region", "#points-join"]
MARKED_ROWS = [["W0 › recto › fiber 1 › kollesis", "1 node"]]
SECOND_ROWS = (f"[...({SECOND_DOCUMENT}?.querySelectorAll('#points-runs button') ?? [])]"
               ".map((row) => [...row.children].map((part) => part.textContent))")
SECOND_FOCUS = f"{SECOND_DOCUMENT}.querySelector('#position-number').value"
CENTRED_S = 5.0
NEAR_VOXELS = 0.5


def stored(value):
    return f"""new Promise((resolve, reject) => {{
  const opening = indexedDB.open("teca-points");
  opening.onerror = () => reject(opening.error);
  opening.onsuccess = () => {{
    const read = opening.result.transaction("skeletons").objectStore("skeletons").get({json.dumps(VOLUME)});
    read.onerror = () => reject(read.error);
    read.onsuccess = () => {{
      opening.result.close();
      resolve(read.result === undefined ? null : ((text) => {value})(read.result));
    }};
  }};
}})"""


STORED = stored('text.split("<node ").length - 1')
STORED_NML = stored("text")


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

    async def click(dx, dy):
        await page.mouse("mouseMoved", x + dx, y + dy)
        await page.mouse("mousePressed", x + dx, y + dy, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", x + dx, y + dy, button="left", buttons=0, clickCount=1)
        await page.wait_text(NOTE, lambda note: note.endswith(KEPT), 60)

    async def second_note(start):
        deadline = time.monotonic() + SECOND_S
        while not (note := await page.evaluate(SECOND_NOTE)).startswith(start) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return note

    async def clear_second():
        await page.evaluate(f"({SECOND_DOCUMENT}.querySelector('#points-clear').click(), true)")
        await asyncio.sleep(CLEARED_S)
        return await page.evaluate(STORED)

    await page.click("#points-tool")
    await click(*CLICKS_PX[0])
    await page.click("#points-mark-kollesis")
    await click(*CLICKS_PX[1])
    await page.click("#points-mark-kollesis")
    await page.evaluate("(() => { const frame = document.createElement('iframe'); frame.id = 'second'; "
                        "frame.src = location.href; frame.style = 'position: fixed; right: 0; bottom: 0; width: 480px; "
                        "height: 360px'; document.body.append(frame); return true })()")
    if not (note := await second_note(WATCHED)).startswith(WATCHED):
        failures.append(f"the second page of the CT says {note!r}, not that another tab has its points")
    off = await page.evaluate(f"{json.dumps(EDITS)}.map((id) => {SECOND_DOCUMENT}.querySelector(id)?.disabled ?? null)")
    if off != [True] * len(EDITS):
        failures.append(f"on the second page {EDITS} are disabled {off}, not all")
    if (rows := await page.evaluate(SECOND_ROWS)) != MARKED_ROWS:
        failures.append(f"the second page's marked runs list {rows}, not {MARKED_ROWS}")
    else:
        nml = await page.evaluate(STORED_NML)
        marked_id = re.search(r'<comment node="(\d+)" content="\[KOLLESIS\]"', nml)[1]
        node = re.search(rf'<node id="{marked_id}"[^>]*>', nml)[0]
        marked = [float(re.search(rf' {axis}="([^"]+)"', node)[1]) for axis in "xyz"]
        await page.evaluate(f"({SECOND_DOCUMENT}.querySelector('#points-runs button').click(), true)")
        await page.wait_text(SECOND_FOCUS, lambda focus: all(
            abs(float(value) - at) <= NEAR_VOXELS for value, at in zip(focus.split(","), marked)), CENTRED_S)
        if not (note := await page.evaluate(SECOND_NOTE)).startswith(WATCHED):
            failures.append(f"after a click on the run's row the second page says {note!r}")
    await click(*CLICKS_PX[2])
    if (kept := await clear_second()) != 3:
        failures.append(f"after Clear all on the second page, IndexedDB keeps {kept} of the first page's 3 nodes")

    await page.reveal("#open-source-button")
    await page.click("#open-source-button")
    await page.type_into("zarr-url", OTHER)
    await page.click("#load-form button[type='submit']")
    note = await second_note(BROUGHT)
    print(f"once the first page opened another volume, the second said {note!r}")
    if note != BROUGHT:
        failures.append(f"once the first page opened another volume, the second page says {note!r}, not {BROUGHT!r}")
    elif (kept := await clear_second()) is not None:
        failures.append(f"Clear all on the second page, which holds the CT, leaves IndexedDB {kept} nodes")
    if (dialogs := len(page.browser.dialogs)) != 1:
        failures.append(f"Clear all asked {dialogs} times, not once, on the page that holds the CT")
    return failures
