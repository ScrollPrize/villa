import time

from page import VOLUME, ct_key

# Runs before the viewer's modules; a creation transaction stays open while it has a request pending.
HOLD_CREATION = """
(() => {
  const request = indexedDB.open("teca.downloads.v1", 1);
  request.onupgradeneeded = () => {
    const store = request.result.createObjectStore("hold");
    const spin = () => {
      store.get(0).onsuccess = spin;
    };
    spin();
  };
})();
"""


async def check(page, out):
    await page.browser.send("Page.addScriptToEvaluateOnNewDocument", {"source": HOLD_CREATION})
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.click("#open-settings")
    status = await page.browser.wait_for("document.querySelector('#download-cache-status').textContent", 30)
    if "did not open" not in status:
        return [f"the settings show the download storage as {status!r}, not as not opened"]
    return []
