import asyncio
import json
import time

from page import BOOT_TIMEOUT_S, VOLUME, ct_key, session_link

SAVED = VOLUME.rstrip("/").split("/")[-1].removesuffix(".zarr") + "-session.json"
NOT_A_SESSION = "not-a-session.json"

UNREAD = "unread"
STATUS = "document.querySelector('#status').textContent"
SIZE = "((session) => new TextEncoder().encode(JSON.stringify(session)).length)"
PUBLISHED_S = 0.5
AGAIN_S = 1.0
SAID_S = 0.5
SAVE_REFUSED_S = 1.0
NOT_SAVED = "Session not saved: the view is past what a session holds."


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await asyncio.sleep(PUBLISHED_S)
    if not (address := await page.address()).startswith("#session="):
        failures.append(f"once the CT opens the address is {address[:60]}…, not #session= alone")
        return failures
    opened = await page.address_session()

    await page.reveal("#save-session")
    saved_path = await page.download("#save-session", SAVED)
    saved = json.loads(saved_path.read_text())["session"]
    if saved != (shown := await page.address_session()):
        differ = [key for key in shown if saved.get(key) != shown[key]]
        failures.append(f"Save session's {SAVED} differs from the address's session in {differ}")
        return failures
    await page.open("")
    before = await page.evaluate("performance.timeOrigin")
    t0 = time.monotonic()
    await page.choose_file("#session-file", saved_path)
    try:
        await page.browser.wait_for(f"performance.timeOrigin !== {before} && window.__tecaDebug !== undefined",
                                    BOOT_TIMEOUT_S)
    except TimeoutError as error:
        failures.append(f"Open session… with {SAVED} left the page with no view: {error}")
        return failures
    await page.settle(t0, {"panes complete": ct_key})
    if (shown := await page.evaluate("window.__tecaDebug.session()")) != saved:
        differ = [key for key in saved if shown.get(key) != saved[key]]
        failures.append(f"Open session… with {SAVED} opens a view whose {differ} differ from the file")
        return failures
    not_a_session = page.browser.downloads / NOT_A_SESSION
    not_a_session.write_text(json.dumps({"session": {**saved, "rotationMode": "euler"}, "kept": {}}))
    before = await page.evaluate("performance.timeOrigin")
    await page.choose_file("#session-file", not_a_session)
    want = f"Session file {NOT_A_SESSION} not opened: Unknown rotation mode: euler."
    said = await page.wait_text(STATUS, lambda text: text.startswith("Session file"), BOOT_TIMEOUT_S)
    if said != want or await page.evaluate("performance.timeOrigin") != before:
        failures.append(f"Open session… with a rotation mode of euler says {said!r} and reloads the page or not, where "
                        f"it should say {want!r} and leave the view")

    limit = await page.config("SESSION_JSON_BYTE_LIMIT")
    session = {**opened, "tabs": {"left": "data", "right": "info"}, "extensions": {**opened["extensions"], UNREAD: ""}}
    filler = limit - await page.evaluate(f"{SIZE}({json.dumps(session)})")
    session["extensions"][UNREAD] = "x" * filler
    before = await page.evaluate("performance.timeOrigin")
    t0 = time.monotonic()
    await page.browser.navigate(page.root + session_link(session))
    try:
        await page.browser.wait_for(f"performance.timeOrigin !== {before} && window.__tecaDebug !== undefined",
                                    BOOT_TIMEOUT_S)
    except TimeoutError as error:
        failures.append(f"the link pasted over the address left the page as it was: {error}")
        return failures
    await page.settle(t0, {"panes complete": ct_key})
    shown = await page.evaluate("window.__tecaDebug.session()")
    held = await page.address_session()
    if shown != session or held != session:
        differ = [[key for key in session if other[key] != session[key]] for other in (shown, held)]
        failures.append(f"the pasted link of {limit} bytes opens a view whose {differ[0]} differ from it, and an address "
                        f"whose {differ[1]} do")
        return failures

    async def black(value):
        await page.choose("#display-black", value)
        said = await page.evaluate(STATUS)
        await asyncio.sleep(PUBLISHED_S)
        return (await page.address_session())["displayBlack"], said

    want = (f"The link holds the view as it last fitted: this one is {limit + 2:,} bytes, over {limit:,}, and {UNREAD} "
            f"take {filler + 2:,} of them.")
    if (found := await black("120")) != (0, want):
        failures.append(f"a black of 120 gives the address's black and the status bar {found}, not 0 and {want!r}")
    kept, said = await black("9")
    if kept != 9 or said == want:
        failures.append(f"a black of 9, back at the limit, leaves the address's black {kept} and the bar saying {said!r}")
    t0 = time.monotonic()
    if (found := await black("120")) != (9, want):
        failures.append(f"a black of 120 again gives the address's black and the status bar {found}, not 9 and {want!r}")
    await asyncio.sleep(AGAIN_S - (time.monotonic() - t0))
    await black("121")
    await asyncio.sleep(t0 + await page.config("STATUS_HOLD_MS") / 1000 + SAID_S - time.monotonic())
    if (said := await page.evaluate(STATUS)) == want:
        failures.append(f"a black of 121, still past the limit, said again what 120 had: {said!r}")

    # A grant denies every permission it leaves out, the clipboard's write among them.
    await page.browser.send("Browser.grantPermissions", {"permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"],
                                                         "origin": page.root.rstrip("/")})
    await page.reveal("#copy-session")
    before = await page.evaluate(STATUS)
    await page.click("#copy-session")
    said = await page.wait_text(STATUS, lambda text: text != before, BOOT_TIMEOUT_S)
    copied = await page.evaluate("navigator.clipboard.readText()")
    address = await page.evaluate("location.href")
    if copied != address or said != f"Session link copied. {want}":
        failures.append(f"Copy session link past the limit copies {copied[:80]}… and says {said!r}, where the address "
                        f"is {address[:80]}…")
    files = sorted(page.browser.downloads.iterdir())
    await page.click("#save-session")
    said = await page.wait_text(STATUS, lambda text: text.startswith("Session not saved"), BOOT_TIMEOUT_S)
    await asyncio.sleep(SAVE_REFUSED_S)
    written = [path.name for path in sorted(page.browser.downloads.iterdir()) if path not in files]
    if written or said != f"{NOT_SAVED} {want}":
        failures.append(f"Save session past the limit says {said!r} and writes {written}, where it should say "
                        f"{NOT_SAVED!r} and write nothing")

    return failures
