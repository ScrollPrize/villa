import json
import re
import time

from page import BOOT_TIMEOUT_S, CENTER_XYZ, VOLUME, ct_key

ZOOM = 4
CLICKS_PX = [(-40, -20), (40, 20), (40, -20)]
KEPT = ", kept in this browser."
SAVED = VOLUME.rstrip("/").split("/")[-1].removesuffix(".zarr") + "-session.json"
NOTE = "document.querySelector('#points-note').textContent"
ASKED = "Replace the 3 nodes here with the session file's nodes?"
BROUGHT = "Brought back 2 nodes placed here before."
STORED_NML = f"""new Promise((resolve, reject) => {{
  const opening = indexedDB.open("teca-points");
  opening.onerror = () => reject(opening.error);
  opening.onsuccess = () => {{
    const read = opening.result.transaction("skeletons").objectStore("skeletons").get({json.dumps(VOLUME)});
    read.onerror = () => reject(read.error);
    read.onsuccess = () => {{
      opening.result.close();
      resolve(read.result ?? null);
    }};
  }};
}})"""


def nodes(nml):
    return None if nml is None else len(re.findall(r"<node ", nml))


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

    async def open_file(path):
        before = await page.evaluate("performance.timeOrigin")
        t0 = time.monotonic()
        await page.choose_file("#session-file", path)
        await page.browser.wait_for(f"performance.timeOrigin !== {before} && window.__tecaDebug !== undefined",
                                    BOOT_TIMEOUT_S)
        await page.settle(t0, {"panes complete": ct_key})
        return await page.wait_text(NOTE, lambda note: note.endswith(KEPT) or note.startswith("Brought back"), 60)

    await page.click("#points-tool")
    await click(*CLICKS_PX[0])
    await click(*CLICKS_PX[1])
    await page.reveal("#save-session")
    path = await page.download("#save-session", SAVED)
    filed = json.loads(path.read_text())["kept"].get("points")
    stored = await page.evaluate(STORED_NML)
    if filed is None or filed["ct"] != VOLUME or filed["nml"] != stored or nodes(stored) != 2:
        failures.append(f"Save session keeps {json.dumps(filed)[:120]} of Points, where IndexedDB keeps "
                        f"{nodes(stored)} nodes of {VOLUME}")
        return failures

    await page.reveal("#points-section")
    await click(*CLICKS_PX[2])
    await open_file(path)
    if page.browser.dialogs != [ASKED] or (kept := nodes(await page.evaluate(STORED_NML))) != 2:
        failures.append(f"the session file opened over 3 nodes asked {page.browser.dialogs}, not {[ASKED]}, and "
                        f"IndexedDB keeps {kept} nodes, not its 2")
        return failures

    said = await open_file(path)
    if page.browser.dialogs != [ASKED] or not said.startswith(BROUGHT):
        failures.append(f"the session file opened over its own 2 nodes asked {page.browser.dialogs[1:]} and the note "
                        f"says {said!r}, where it should ask nothing and start {BROUGHT!r}")
    return failures
