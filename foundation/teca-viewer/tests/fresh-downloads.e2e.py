import asyncio
import time

from page import POLL_S, VOLUME, VOLUME_PATH, ct_key

FRESH_WAIT_S = 10.0
ENTRIES = """((until) => new Promise((resolve, reject) => {
  const opening = indexedDB.open("teca.downloads.v1", 1);
  opening.onerror = () => reject(opening.error);
  opening.onsuccess = () => {
    const database = opening.result;
    const transaction = database.transaction("entries", until === null ? "readonly" : "readwrite");
    const counts = { entries: 0, fresh: 0 };
    transaction.objectStore("entries").openCursor().onsuccess = ({ target }) => {
      const cursor = target.result;
      if (cursor === null) return;
      counts.entries++;
      if (until !== null) cursor.update({ ...cursor.value, freshUntil: until });
      else if (cursor.value.freshUntil > Date.now()) counts.fresh++;
      cursor.continue();
    };
    transaction.oncomplete = () => {
      database.close();
      resolve(counts);
    };
    transaction.onerror = () => reject(transaction.error);
  };
}))"""


def chunk_requests(browser, t0):
    counts = {}
    for r in browser.requests.values():
        name = r["url"].rsplit("/", 1)[-1]
        if r["start"] >= t0 and VOLUME_PATH in r["url"] and not name.startswith((".z", "zarr.json")):
            counts[r["method"]] = counts.get(r["method"], 0) + 1
    return counts


async def check(page, out):
    failures = []
    await page.open(f"?volume={VOLUME}")
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    href = await page.address()

    async def reload():
        t0 = time.monotonic()
        await page.open(href)
        times, _ = await page.settle(t0, {"panes complete": ct_key})
        return times["panes complete"], chunk_requests(page.browser, t0)

    seconds, asked = await reload()
    stored = await page.evaluate(f"{ENTRIES}(null)")
    print(f"the warm reload completed its panes in {seconds:.2f} s asking {asked} of {stored['entries']} stored chunks")
    if stored["entries"] == 0 or stored["fresh"] != stored["entries"]:
        failures.append(f"of the {stored['entries']} chunks stored by the open, {stored['fresh']} are fresh")
    if asked:
        failures.append(f"the warm reload asked {asked} of the CT's chunks, not nothing")

    await page.evaluate(f"{ENTRIES}(0)")
    seconds, asked = await reload()
    print(f"with every entry stale, the reload completed its panes in {seconds:.2f} s asking {asked}")
    if set(asked) != {"HEAD"}:
        failures.append(f"with every stored chunk stale, the reload asked {asked}, not a HEAD of each chunk alone")
    checked = asked.get("HEAD", 0)
    t0 = time.monotonic()
    while (stored := await page.evaluate(f"{ENTRIES}(null)"))["fresh"] < checked:
        if time.monotonic() - t0 > FRESH_WAIT_S:
            failures.append(f"{FRESH_WAIT_S:g} s after the reload, {stored['fresh']} of the {checked} chunks checked are fresh")
            break
        await asyncio.sleep(POLL_S)
    seconds, asked = await reload()
    print(f"once they were checked, the reload completed its panes in {seconds:.2f} s asking {asked}")
    if asked:
        failures.append(f"the reload after the check asked {asked} of the CT's chunks, not nothing")
    return failures
