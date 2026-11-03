import asyncio
import json
import time

from page import POLL_S, SURFACE, TIMEOUT_S, VOLUME, WIDTH, ct_key, lines_key

SEARCH = "paris4 2.400um 78kev"
SEGMENT = "20231221180251"
EVERY_SEARCH = f"{SEGMENT} 2.4 µm"
TITLE = "PHercParis4 20260411134726-2.400um-0.2m-78keV-masked"
FIELDS = ["2.4 µm", "78 keV", "2026-04-11"]
ROWS = [["7.91 µm", "54 keV", "2023-02-05"], ["45.532 µm", "74 keV", "2026-03-10"], FIELDS,
        ["1.129 µm", "78 keV", "2026-06-08"]]
KEYS = ["ArrowDown", "ArrowDown", "ArrowDown", "ArrowUp", "ArrowDown"]
CATALOG_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json"
REVALIDATED_BYTES = 4096   # a 304 is headers alone; the catalog is 1.3 MB
HEIGHT = 700
TALL = 1600
LIST_ROWS, ROW_PX = 5, 34
LISTS = """(() => {
  const body = document.querySelector('#left-panel .panel-body');
  const foot = body.getBoundingClientRect().bottom - parseFloat(getComputedStyle(body).paddingBottom);
  const r = document.querySelector('#left-panel').getBoundingClientRect();
  return [['#catalog-volumes', '#catalog-surfaces'].map((id) => document.querySelector(id).clientHeight / %d),
    document.querySelector('[data-lpanel="data"]').getBoundingClientRect().bottom - foot,
    { x: r.x, y: r.y, width: r.width, height: r.height }];
})()""" % ROW_PX
SCROLLS = """(() => {
  const list = document.querySelector('#catalog-volumes').getBoundingClientRect();
  const row = document.querySelector('#catalog-volumes [aria-selected="true"]')?.getBoundingClientRect();
  return [document.querySelector('#left-panel .panel-body').scrollTop, row !== undefined && row.top >= list.top && row.bottom <= list.bottom];
})()"""
FOCUSED = "document.activeElement.id || document.activeElement.tagName.toLowerCase()"
GRID = ["3d", "xy", "xz", "zy"]
ROW_STATE = """[...document.querySelectorAll({} + ' .catalog-row')].map((row) => {{
  const [first, second] = [...row.children].map((line) => [...line.children].map((span) => span.textContent));
  const fits = [...row.children].every((line) => line.lastChild.getBoundingClientRect().right <= line.getBoundingClientRect().right);
  return {{ url: row.dataset.url, first, fields: second, title: row.title, chosen: row.getAttribute('aria-selected') === 'true',
    background: getComputedStyle(row).backgroundColor, fits }};
}})"""


async def check(page, out):
    failures = []

    async def listed(t0):
        while await page.evaluate("document.querySelector('#catalog-lists').hidden"):
            if time.monotonic() - t0 > TIMEOUT_S:
                note = await page.evaluate("document.querySelector('#catalog-note').textContent")
                raise TimeoutError(f"the catalog did not load in {TIMEOUT_S} s: {note}")
            await asyncio.sleep(POLL_S)

    async def rows(selector):
        return await page.evaluate(ROW_STATE.format(json.dumps(selector)))

    async def scope():
        return await page.evaluate("document.querySelector('#catalog-surface-scope').value")

    await page.open("")
    await page.browser.send("Emulation.setDeviceMetricsOverride", {"width": WIDTH, "height": HEIGHT, "deviceScaleFactor": 1,
                                                                    "mobile": False})
    if (place := await page.place("#catalog-section")) != "data":
        failures.append(f"its card is on {place!r}, not the Data tab")
    await page.reveal("#catalog-section")
    t0 = time.monotonic()
    await page.click("#catalog-load")
    await listed(t0)

    if await scope() != "all":
        failures.append(f"with no volume open the surfaces listed are of {await scope()!r}, not every volume's")
    await page.type_into("catalog-surface-search", SEGMENT)
    found = await rows("#catalog-surfaces")
    if [[row["first"], row["fields"]] for row in found] != [[[SEGMENT, "PHercParis4"], fields] for fields in ROWS]:
        failures.append(f"the rows of {SEGMENT} read {[[row['first'], row['fields']] for row in found]}, not {ROWS}")
        return failures
    if unfit := [row["fields"] for row in found if not row["fits"]]:
        failures.append(f"the panel's width does not hold the rows {unfit} on two lines")
    if found[2]["url"] != SURFACE or found[2]["title"] != TITLE:
        failures.append(f"the third row is {found[2]['url']} titled {found[2]['title']!r}, not {SURFACE} titled {TITLE!r}")
    for name in KEYS:
        await page.key(name)
    found = await rows("#catalog-surfaces")
    if [row["chosen"] for row in found] != [False, False, True, False]:
        failures.append(f"after {KEYS} the rows chosen are {[row['chosen'] for row in found]}, not the third alone")
    if len({row["background"] for row in found}) != 2:
        failures.append(f"the chosen row's background is not the others' alone: {[row['background'] for row in found]}")
    await page.type_into("catalog-surface-search", EVERY_SEARCH)
    if [[row["url"], row["chosen"]] for row in await rows("#catalog-surfaces")] != [[SURFACE, True]]:
        failures.append(f"searching {EVERY_SEARCH!r} lists {await rows('#catalog-surfaces')}, not {SURFACE} chosen")
        return failures
    x, y = await page.evaluate("(() => { const r = document.querySelector('#left-panel .panel-body').getBoundingClientRect(); "
                               "return [r.x + r.width / 2, r.y + 20] })()")
    await page.mouse("mouseWheel", x, y, deltaX=0, deltaY=-HEIGHT)
    await page.browser.wait_for("document.querySelector('#left-panel .panel-body').scrollTop === 0", 10)
    t0 = time.monotonic()
    await page.key("Enter")
    settled, _ = await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    print(f"Enter opened the volume and drew its panes in {settled['panes complete']} s, the segment's lines in "
          f"{settled['lines settled']} s")
    if (opened := await page.evaluate("document.querySelector('#zarr-url').value")) != VOLUME:
        failures.append(f"adding the surface opened {opened}")
    scrolled, shown = await page.evaluate(SCROLLS)
    if scrolled != 0:
        failures.append(f"opening the volume scrolled the panel {scrolled} px to its row in the Volumes list")
    if not shown:
        failures.append("the Volumes list does not show the open volume's row")

    await page.key("2")
    await page.key("z")
    focused, panes = await page.evaluate(FOCUSED), sorted(await page.panes())
    if focused != "body" or panes != ["zy"]:
        failures.append(f"after Enter, 2 and Z show {panes} with the focus on {focused}, not YZ alone on the page")
        return failures
    await page.key("z")
    for selector, next_id in [("#catalog-volumes", "catalog-open"), ("#catalog-surfaces", "catalog-add")]:
        focus = f"document.querySelector({json.dumps(selector)}).focus(), true"
        await page.evaluate(focus)
        await page.key("1")
        await page.key("z")
        kept = sorted(await page.panes())
        await page.key("Escape")
        blurred = await page.evaluate(FOCUSED)
        await page.key("z")
        freed = sorted(await page.panes())
        await page.key("z")
        await page.evaluate(focus)
        await page.key("Tab")
        got = [kept, blurred, freed, await page.evaluate(FOCUSED)]
        if got != (want := [GRID, "body", ["zy"], next_id]):
            failures.append(f"in the focused {selector} 1 and Z show {got[0]}, Escape leaves the focus on {got[1]}, Z then "
                            f"shows {got[2]} and Tab moves the focus to {got[3]}, not {want}")
    await page.evaluate("document.activeElement.blur(), true")

    await page.choose("#catalog-surface-scope", "open")
    await page.type_into("catalog-surface-search", SEGMENT)
    if [[row["url"], row["fields"]] for row in await rows("#catalog-surfaces")] != [[SURFACE, ROWS[2]]]:
        failures.append(f"the volume's surfaces matching {SEGMENT} are {await rows('#catalog-surfaces')}, not {SURFACE}")
    await page.type_into("catalog-search", SEARCH)
    volumes = await rows("#catalog-volumes")
    if [[row["url"], row["first"], row["fields"], row["title"]] for row in volumes] != [[VOLUME, ["PHercParis4"], FIELDS, TITLE]]:
        failures.append(f"searching {SEARCH!r} lists {volumes}, not the real scroll's volume")
        return failures
    await page.click(f"#catalog-volumes [data-url={json.dumps(VOLUME)}]")
    t0 = time.monotonic()
    await page.key("Enter")
    await page.settle(t0, {"panes complete": ct_key})
    if (opened := await page.evaluate("document.querySelector('#zarr-url').value")) != VOLUME:
        failures.append(f"opening it opened {opened}")
    await page.key("z")
    focused, panes = await page.evaluate(FOCUSED), sorted(await page.panes())
    if focused != "body" or len(panes) != 1:
        failures.append(f"after Enter on the volume's row Z shows {panes} with the focus on {focused}, not one pane")
    await page.key("z")

    t0 = time.monotonic()
    await page.open(await page.address())
    await listed(t0)
    await page.settle(t0, {"panes complete": ct_key})
    searches = await page.evaluate("['#catalog-search', '#catalog-surface-search', '#catalog-surface-scope']"
                                   ".map((id) => document.querySelector(id).value)")
    if searches != [SEARCH, SEGMENT, "open"]:
        failures.append(f"the reloaded page searches {searches}, not {[SEARCH, SEGMENT, 'open']}")
    loads = [request for request in page.browser.requests.values() if request["url"] == CATALOG_URL]
    if len(loads) != 2 or loads[1]["bytes"] > REVALIDATED_BYTES:
        failures.append(f"the catalog was fetched as {loads}; the reloaded page's fetch should be a 304")

    for search in ["#catalog-search", "#catalog-surface-search"]:
        await page.choose(search, "")
    for height in [HEIGHT, TALL]:
        await page.browser.send("Emulation.setDeviceMetricsOverride", {"width": WIDTH, "height": height,
                                                                        "deviceScaleFactor": 1, "mobile": False})
        await page.evaluate("document.querySelector('#catalog-section').scrollIntoView(), new Promise((resolve) => "
                            "requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        shown, past, rect = await page.evaluate(LISTS)
        (out / f"lists-{height}.png").write_bytes(await page.browser.screenshot(rect))
        print(f"on a page {height} px tall the lists show {shown[0]:.1f} and {shown[1]:.1f} rows")
        if min(shown) < LIST_ROWS or (height == TALL and (min(shown) < LIST_ROWS + 1 or abs(past) > 1)):
            failures.append(f"on a page {height} px tall the lists show {shown} rows and the Data tab ends {past:.0f} px "
                            f"past the panel's foot, {out / f'lists-{height}.png'}")
    return failures
