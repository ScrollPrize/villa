import asyncio
from urllib.parse import quote


def module(name):
    return (f"data:text/javascript,(window.consentRan ??= []).push('{name}');"
            f" export default () => {{ const panel = document.createElement('section'); panel.id = 'consent-{name}';"
            " return { panel, draw() {}, dispose() {} } }")


NAMES = ["skipped", "loaded", "escaped"]
HASH = "#ext=extensions/catalog.js&" + "&".join(f"ext={quote(module(name), safe='')}" for name in NAMES)


async def check(browser):
    failures = []
    asked = []
    for name in NAMES:
        await browser.wait_for(f"document.getElementById('extension-dialog').open && document.getElementById('extension-url').textContent.includes('consent-{name}')", 30)
        asked.append(await browser.evaluate("({ ran: window.consentRan ?? [], focus: document.activeElement.id,"
                                            " catalog: [...document.getElementById('extensions').children].some(section => section.textContent.includes('Catalog')) })"))
        if name == "escaped":
            await browser.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27})
            await browser.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27})
        else:
            await browser.evaluate(f"document.getElementById('extension-{'load' if name == 'loaded' else 'skip'}').click()")
    await browser.wait_for("document.getElementById('extensions-loaded').textContent.split('ext skipped').length === 3", 30)
    await asyncio.sleep(0.2)
    after = await browser.evaluate("({ ran: window.consentRan, open: document.getElementById('extension-dialog').open,"
                                   " panels: ['skipped', 'loaded', 'escaped'].map(name => document.getElementById('consent-' + name) !== null),"
                                   " header: document.getElementById('extensions-loaded').textContent,"
                                   " red: document.getElementById('extensions-loaded').classList.contains('error'),"
                                   " keys: new URLSearchParams(location.hash.slice(1)).getAll('ext').length })")
    if asked != [{"ran": [], "focus": "extension-skip", "catalog": True}, {"ran": [], "focus": "extension-skip", "catalog": True},
                 {"ran": ["loaded"], "focus": "extension-skip", "catalog": True}]:
        failures.append(f"at each question: {asked}")
    if (after["ran"] != ["loaded"] or after["open"] or after["panels"] != [False, True, False] or after["red"] or after["keys"] != 4
            or not after["header"].startswith("ext: catalog.js, ") or "consent-skipped" not in after["header"] or "consent-escaped" not in after["header"]):
        failures.append(f"after the answers: {after}")
    return failures
