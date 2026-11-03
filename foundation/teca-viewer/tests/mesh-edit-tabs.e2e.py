import asyncio
import json
import time

from page import CENTER_XYZ, TIMEOUT_S, VOLUME, ct_key, lines_key

ZOOM = 4
KEPT = ", kept in this browser."
SECOND_S = 60
DISCARDED_S = 1.0
OTHER = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/"
         "ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/")
NOTE = "document.querySelector('#mesh-edit-note').textContent"
SECOND_DOCUMENT = "document.querySelector('#second')?.contentDocument"
SECOND_NOTE = f"({SECOND_DOCUMENT}?.querySelector('#mesh-edit-note')?.textContent ?? '')"
ELSEWHERE = "Another tab edits this surface; it shows here as that tab keeps it, to edit once it lets go of it."
BROUGHT = "Brought back the edits kept here before; Discard forgets them."
STORED = f"""new Promise((resolve, reject) => {{
  const opening = indexedDB.open("teca-mesh-edit");
  opening.onerror = () => reject(opening.error);
  opening.onsuccess = () => {{
    const reading = opening.result.transaction(["edits", "surfaces"]);
    const range = IDBKeyRange.bound([{json.dumps(VOLUME)}], [{json.dumps(VOLUME)}, []]);
    const edits = reading.objectStore("edits").count(range);
    const surfaces = reading.objectStore("surfaces").getAll(range);
    reading.onerror = () => reject(reading.error);
    reading.oncomplete = () => {{
      opening.result.close();
      resolve([edits.result, surfaces.result.map((surface) => surface.done)]);
    }};
  }};
}})"""


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.reveal("#mesh-edit-section")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", str(ZOOM))
    await page.settle(t0, {"panes complete": ct_key})
    rect = (await page.panes())["xy"]
    x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)

    async def kept():
        await page.wait_text(NOTE, lambda note: note.endswith(KEPT), 60)

    async def second_note(text):
        deadline = time.monotonic() + SECOND_S
        while (note := await page.evaluate(SECOND_NOTE)) != text and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return note

    async def discard_second():
        await page.evaluate(f"({SECOND_DOCUMENT}.querySelector('#mesh-edit-discard').click(), true)")
        await asyncio.sleep(DISCARDED_S)
        return await page.evaluate(STORED)

    await page.add_surface()
    await page.settle(time.monotonic(), {"panes complete": ct_key, "lines settled": lines_key})
    await page.click("#mesh-edit-open")
    await page.wait_text(NOTE, lambda note: " vertices, " in note, TIMEOUT_S)
    await page.click("#mesh-edit-erase")
    await page.mouse("mouseMoved", x, y)
    await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
    await page.mouse("mouseReleased", x, y, button="left", buttons=0, clickCount=1)
    await kept()
    await page.evaluate("(() => { const frame = document.createElement('iframe'); frame.id = 'second'; "
                        "frame.src = location.href; frame.style = 'position: fixed; right: 0; bottom: 0; width: 480px; "
                        "height: 360px'; document.body.append(frame); return true })()")
    if (note := await second_note(ELSEWHERE)) != ELSEWHERE:
        failures.append(f"the second page of the CT says {note!r}, not that another tab edits the surface")
    await page.click("#mesh-edit-undo")
    await kept()
    if (stored := await discard_second()) != [1, [0]]:
        failures.append(f"after Discard on the second page, IndexedDB keeps {stored[0]} edits, {stored[1]} done, not "
                        "the first page's erase, undone")

    await page.reveal("#open-source-button")
    await page.click("#open-source-button")
    await page.type_into("zarr-url", OTHER)
    await page.click("#load-form button[type='submit']")
    note = await second_note(BROUGHT)
    print(f"once the first page opened another volume, the second said {note!r}")
    if note != BROUGHT:
        failures.append(f"once the first page opened another volume, the second page says {note!r}, not {BROUGHT!r}")
    elif (stored := await discard_second()) != [0, []]:
        failures.append(f"Discard on the second page, which edits the surface, leaves IndexedDB {stored[0]} edits")
    if (dialogs := len(page.browser.dialogs)) != 1:
        failures.append(f"Discard asked {dialogs} times, not once, on the page that edits the surface")
    return failures
