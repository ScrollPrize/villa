import asyncio
import json
import time

from page import CENTER_XYZ, TIMEOUT_S, VOLUME, ct_key

STROKE_PX = 120
STEP_PX = 10
SECOND_S = 30
READY = "!document.querySelector('#paint-tool').disabled"
NOTE = "document.querySelector('#paint-note').textContent"
SECOND_DOCUMENT = "document.querySelector('#second')?.contentDocument"
SECOND_NOTE = f"({SECOND_DOCUMENT}?.querySelector('#paint-note')?.textContent ?? '')"
BUTTONS = ["#paint-tool", "#paint-clear", "#paint-save", "#paint-load"]
UNRESTORED = "The labels this browser keeps for this CT could not be brought back: "
NEWER = "is less than the existing version (2)"
KEPT_HERE = "Another tab asked for this CT's labels, but IndexedDB has not written an edit of them"
TOOK = "Another tab took this CT's labels; Paint here takes them back."
REQUEST = ("const request = (r) => new Promise((resolve, reject) => { r.onsuccess = () => resolve(r.result); "
           "r.onerror = () => reject(r.error) });")
SEED = f"""(async ([url, edge, good, bad]) => {{
  {REQUEST}
  const opening = indexedDB.open("teca-paint", 1);
  opening.onupgradeneeded = () => {{
    for (const store of ["chunks", "edits", "heads", "journal"]) opening.result.createObjectStore(store);
  }};
  const database = await request(opening);
  const gzip = async (bytes) =>
    new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer());
  const labels = new Uint8Array(edge ** 3).fill(1);
  const [whole, row] = await Promise.all([gzip(labels), gzip(labels.subarray(0, edge))]);
  const writing = database.transaction("chunks", "readwrite");
  writing.objectStore("chunks").put(whole, [url, good]);
  writing.objectStore("chunks").put(row, [url, bad]);
  await new Promise((resolve, reject) => {{ writing.oncomplete = resolve; writing.onabort = () => reject(writing.error) }});
  database.close();
  return true;
}})"""
COUNTS = f"""(async () => {{
  {REQUEST}
  const database = await request(indexedDB.open("teca-paint"));
  const reading = database.transaction(["chunks", "journal", "edits"], "readonly");
  const counts = await Promise.all(["chunks", "journal", "edits"].map((name) => request(reading.objectStore(name).count())));
  database.close();
  return counts;
}})()"""
NEWER_VERSION = f"(async () => {{ {REQUEST} (await request(indexedDB.open('teca-paint', 2))).close(); return true }})()"
DELETE = f"(async () => {{ {REQUEST} await request(indexedDB.deleteDatabase('teca-paint')); return true }})()"


async def check(page, out):
    failures = []

    async def opened():
        t0 = time.monotonic()
        await page.open(f"?volume={VOLUME}")
        await page.settle(t0, {"panes complete": ct_key})
        await page.reveal("#paint-section")

    async def stroke():
        rect = (await page.panes())["xy"]
        x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
        await page.mouse("mouseMoved", x - STROKE_PX // 2, y)
        await page.mouse("mousePressed", x - STROKE_PX // 2, y, button="left", buttons=1, clickCount=1)
        for step_x in range(x - STROKE_PX // 2 + STEP_PX, x + STROKE_PX // 2 + 1, STEP_PX):
            await page.mouse("mouseMoved", step_x, y, button="left", buttons=1)
        await page.mouse("mouseReleased", x + STROKE_PX // 2, y, button="left", buttons=0, clickCount=1)

    async def notes(one, other=None):
        deadline = time.monotonic() + SECOND_S
        while True:
            texts = [await page.evaluate(NOTE), await page.evaluate(SECOND_NOTE)]
            if any(text.startswith(start) for text, start in zip(texts, [one, other]) if start) or time.monotonic() > deadline:
                return texts
            await asyncio.sleep(0.1)

    await opened()
    edge = await page.config("BRICK_EDGE")
    x, y, z = (int(float(value) // edge) for value in CENTER_XYZ.split(","))
    good, bad = f"0/{z}/{y}/{x}", f"0/{z}/{y}/{x + 1}"
    undecoded = f"A chunk of {edge}×{edge}×{edge} voxels got {edge}."
    await page.evaluate(f"{SEED}({json.dumps([VOLUME, edge, good, bad])})")
    await opened()
    note = await page.wait_text(NOTE, lambda note: note.startswith(UNRESTORED), 60)
    if note != f"{UNRESTORED}{undecoded} Reload to try again.":
        failures.append(f"the restore that failed says {note!r}, not why and to reload")
    if (layers := (await page.state())["layers"]) != []:
        failures.append(f"the restore that failed leaves {len(layers)} layer drawn")
    if taken := [button for button in BUTTONS if not await page.evaluate(f"document.querySelector('{button}').disabled")]:
        failures.append(f"the restore that failed leaves {taken} to take")
    await page.click("#paint-tool")
    await stroke()
    await asyncio.sleep(1)
    if (counts := await page.evaluate(COUNTS)) != [2, 0, 0]:
        failures.append(f"after the restore failed, a stroke leaves IndexedDB with {counts} chunks, entries and edits, "
                        "not the 2 chunks it held")

    await page.evaluate(DELETE)
    await opened()
    await page.browser.wait_for(READY, TIMEOUT_S)
    await page.evaluate("(() => { const frame = document.createElement('iframe'); frame.id = 'second'; "
                        "frame.src = location.href; frame.style = 'position: fixed; right: 0; bottom: 0; width: 480px; "
                        "height: 360px'; document.body.append(frame); return true })()")
    if not (second := (await notes(None, "Another tab"))[1]).startswith("Another tab has"):
        failures.append(f"a second page of the CT says {second!r}, not that another tab has its labels")
        return failures
    await page.evaluate(NEWER_VERSION)
    await page.click("#paint-tool")
    await stroke()
    await page.wait_text(NOTE, lambda note: note.endswith(", not kept in this browser."), 60)
    await page.evaluate(f"{SECOND_DOCUMENT}.querySelector('#paint-here').click(), true")
    note, second = await notes(KEPT_HERE, "Brought back")
    print(f"asked for the labels while IndexedDB refuses them, this page says {note!r} and the second {second!r}")
    if not note.startswith(KEPT_HERE) or not second.startswith("Asked the other tab"):
        failures.append(f"asked for its labels while IndexedDB refuses them, this page says {note!r} and the second "
                        f"{second!r}, not that they stay here")
    if len((await page.state())["layers"]) != 1 or not await page.evaluate(READY):
        failures.append("the labels that stay here are not drawn, or the brush is not to take")

    await page.evaluate(DELETE)
    await page.evaluate(f"{SECOND_DOCUMENT}.querySelector('#paint-here').click(), true")
    note, second = await notes(None, "Brought back")
    print(f"asked again once the database is gone, this page says {note!r} and the second {second!r}")
    if note != TOOK or not second.startswith("Brought back") or (await page.state())["layers"]:
        failures.append(f"asked again once the database is gone, this page says {note!r} and the second {second!r}, "
                        "not that it took the labels and brought them back")
    if page.expected(undecoded) == 0 or page.expected(NEWER) == 0:
        failures.append(f"the console holds no {undecoded!r} or no {NEWER!r}")
    return failures
