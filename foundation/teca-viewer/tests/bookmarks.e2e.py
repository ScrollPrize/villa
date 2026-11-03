import asyncio
import time

from page import CENTER_XYZ, POLL_S, TIMEOUT_S, VOLUME, ct_key

OTHER = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/"
         "ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/")
LABEL = "start of the fold"
PASTE = ["20400.5, 13900.25, 43800", "20600, 14100.125, 44000 - fiber", "20500, 14000, 43950 - void - surface break"]
PASTED = [["", "20400.5, 13900.25, 43800"], ["fiber", "20600, 14100.125, 44000"],
          ["void - surface break", "20500, 14000, 43950"]]
REFUSED = [(["20400, 13900, 43800", "kollesis end", "20500, 14000, 43900"], 'Line 2, "kollesis end", is not x, y, z.'),
           (["20400, 13900, 43800", "20400, 13900, 99999 - past the end"],
            'Line 2, "20400, 13900, 99999 - past the end", lies outside the CT')]
CLICKED = 3
DELETED = 2
LONG = 40
ROWS = ("[...document.querySelectorAll('#bookmarks-list .bookmarks-go')]"
        ".map((row) => [...row.children].map((piece) => piece.textContent))")
FIELD = "document.querySelector('#bookmarks-text').value"
NOTE = "document.querySelector('#bookmarks-note').textContent"
FOCUS = "document.querySelector('#position-number').value"


async def check(page, out):
    failures = []
    await page.open("")
    off = await page.evaluate("['#bookmarks-text', '#bookmarks-add'].map((id) => document.querySelector(id).disabled)")
    if off != [True, True]:
        failures.append(f"with no CT open, the field and Add are disabled {off}, not both")
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    if (place := await page.place("#bookmarks-section")) != "Bookmarks":
        failures.append(f"its panel is on {place!r}, not the Bookmarks tab")
    await page.reveal("#bookmarks-section")
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    focus = await page.evaluate(FOCUS)

    async def saved():
        return (await page.address_session())["extensions"].get("bookmarks")

    async def stored(count):
        t0 = time.monotonic()
        while len((await saved() or {}).get(VOLUME, [])) != count and time.monotonic() - t0 < TIMEOUT_S:
            await asyncio.sleep(POLL_S)
        return len((await saved() or {}).get(VOLUME, []))

    async def add(lines, enter=True):
        await page.type_into("bookmarks-text", "\n".join(lines))
        if enter:
            await page.key("Enter")
        else:
            await page.click("#bookmarks-add")
        return await page.evaluate(f"[{ROWS}, {FIELD}, {NOTE}]")

    async def opened(url):
        t0 = time.monotonic()
        await page.reveal("#open-source-button")
        await page.click("#open-source-button")
        await page.type_into("zarr-url", url)
        await page.click("#load-form button[type='submit']")
        await page.settle(t0, {"panes complete": ct_key})
        await page.reveal("#bookmarks-section")
        return await page.evaluate(ROWS)

    rows, field, _ = await add([LABEL], enter=False)
    if [rows, field] != [want := [[LABEL, focus]], ""]:
        failures.append(f"Add with {LABEL!r} at the focus {focus} lists {rows} and leaves {field!r}, not {want} and ''")
    rows, field, _ = await add(PASTE)
    if [rows, field] != [want := [[LABEL, focus], *PASTED], ""]:
        failures.append(f"a paste of {PASTE} lists {rows} and leaves {field!r}, not {want} and ''")
    for lines, said in REFUSED:
        refused, field, note = await add(lines)
        if refused != rows or field != "\n".join(lines) or not note.startswith(said):
            failures.append(f"a paste of {lines} lists {refused}, leaves {field!r} and says {note!r}: it should list "
                            f"{rows}, keep the paste and say {said!r}")

    await page.click(f"#bookmarks-list .bookmarks-row:nth-child({CLICKED}) .bookmarks-go")
    went = await page.evaluate(f"[{FOCUS}, [...document.querySelectorAll('#bookmarks-list .bookmarks-go')]"
                               ".map((row) => row.classList.contains('active'))]")
    if went != [rows[CLICKED - 1][1], [i == CLICKED - 1 for i in range(len(rows))]]:
        failures.append(f"a click on row {CLICKED}, {rows[CLICKED - 1]}, gives the position field and the rows marked "
                        f"{went}")
    await page.click(f"#bookmarks-list .bookmarks-row:nth-child({DELETED}) button[title=Delete]")
    kept = rows[:DELETED - 1] + rows[DELETED:]
    if (rows := await page.evaluate(ROWS)) != kept:
        failures.append(f"Delete on row {DELETED} lists {rows}, not {kept}")

    if (count := await stored(len(kept))) != len(kept):
        failures.append(f"after the Delete the address keeps {count} bookmarks, not {len(kept)}")
    t0 = time.monotonic()
    await page.open(await page.address())
    await page.settle(t0, {"panes complete": ct_key})
    if (rows := await page.evaluate(ROWS)) != kept:
        failures.append(f"the page opened from the address lists {rows}, not {kept}")

    if (rows := await opened(OTHER)) != []:
        failures.append(f"another CT lists {rows}, not none")
    if len((held := await saved() or {}).get(VOLUME, [])) != len(kept) or OTHER in held:
        failures.append(f"while another CT is open the address keeps {held}, not the {len(kept)} bookmarks of the first")
    if (rows := await opened(VOLUME)) != kept:
        failures.append(f"the first CT opened again lists {rows}, not {kept}")

    lines = [f"{20000 + row}, 14000, 43900 - row {row}" for row in range(LONG)]
    rows, _, _ = await add(lines)
    scrolls = await page.evaluate("[document.querySelector('#bookmarks-list'), "
                                  "document.querySelector('#bookmarks-section').closest('.panel-body')]"
                                  ".map((box) => box.scrollHeight > box.clientHeight)")
    if len(rows) != len(kept) + LONG or scrolls != [True, False]:
        failures.append(f"with {len(rows)} rows, [the list, the tab] scroll {scrolls}, not the list alone")

    # A bookmark of such a line is 66 bytes of JSON, so these pass the limit by half.
    over = [f"{20000 + row % LONG}.125, 14000.125, 43900.125 - row {row}" for row in range(
        await page.config("SESSION_JSON_BYTE_LIMIT") // 40)]
    refused, _, note = await add(over)
    count = await stored(len(rows))
    await page.click("#bookmarks-list .bookmarks-row:nth-child(1) .bookmarks-go")
    added, _, _ = await add([LABEL])
    want = [rows, len(rows), rows + [[LABEL, rows[0][1]]]]
    if [refused, count, added] != want or note != f"The link cannot hold {len(over)} more bookmarks. Nothing was added.":
        failures.append(f"a paste of {len(over)} lines past the address's limit lists {len(refused)} rows, leaves "
                        f"{count} in the address and says {note!r}, and a label at row 1 then lists {added[len(rows):]}: "
                        f"it should list and leave {len(rows)}, say so, and add the label at {rows[0][1]}")
    return failures
