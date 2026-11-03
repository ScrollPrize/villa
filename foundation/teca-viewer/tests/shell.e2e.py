import asyncio
import math
import time

from page import CENTER_XYZ, SHIFT, VOLUME, ct_key, lines_key, surface_key

HEIGHT_PX = 1
HELD_S = 0.3
TURNS = 30
ROUNDED = 0.002   # the position field writes three decimals, so a step along a turned normal is 1 within this
FOCUS = "document.querySelector('#position-number').value"
PUBLISHED_S = 0.5
LEFT_CARDS = {"data": ["Volume", "Surfaces"], "view": ["Position", "Rotation"]}
INFO_CARDS = ["Source metadata", "Performance"]
TURNED_S = 0.5
HIDDEN = "document.querySelector('.viewer-shell').classList.contains('crosshair-hidden')"
STATUS = "document.querySelector('#status').textContent"
ZOOM = "document.querySelector('#zoom').value"
ROTATION = "document.querySelector('#rotation-text').textContent"
SHOWN = """(() => {   // each side's tab marked active and the panels shown, each with its cards' titles and height
  const side = (tab, panel) => ({
    tab: document.querySelector(`.${tab}.active`)?.dataset[tab],
    panels: [...document.querySelectorAll(`.${panel}:not([hidden])`)].map((shown) => ({
      name: shown.dataset[panel],
      cards: [...shown.querySelectorAll(".card")].map((card) => card.querySelector("h2")?.textContent.trim()),
      height: shown.getBoundingClientRect().height,
      body: shown.parentElement.clientHeight - parseFloat(getComputedStyle(shown.parentElement).paddingTop) -
        parseFloat(getComputedStyle(shown.parentElement).paddingBottom),
    })),
  });
  return { left: side("ltab", "lpanel"), right: side("rtab", "rpanel"),
           rtabs: [...document.querySelectorAll(".rtab")].map((tab) => [tab.dataset.rtab, tab.title]) };
})()"""


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})

    async def moved(press):
        before = [float(value) for value in (await page.evaluate(FOCUS)).split(",")]
        await press()
        return [round(float(value) - start, 3) for value, start in zip((await page.evaluate(FOCUS)).split(","), before)]

    async def held():
        await page.key("F", down_only=True, shift=True)
        await asyncio.sleep(HELD_S)
        await page.browser.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "F", "code": "KeyF", "modifiers": SHIFT,
                                                           "windowsVirtualKeyCode": 70})

    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.key("1")
    steps = [await moved(lambda: page.key("F", shift=True)), await moved(lambda: page.key("D", shift=True)),
             await moved(held)]
    if steps != [[0, 0, 1], [0, 0, -1], [0, 0, 1]]:
        failures.append(f"on XY Shift+F, Shift+D and Shift+F held {HELD_S} s move the focus XYZ by {steps}, not one "
                        f"slice up, down and up")
    for _ in range(TURNS):
        await page.key("ArrowRight")
    turned = await moved(lambda: page.key("F", shift=True))
    for _ in range(TURNS):
        await page.key("ArrowLeft")
    if abs(math.hypot(*turned) - 1) > ROUNDED or sorted(abs(value) for value in turned)[1] < ROUNDED:
        failures.append(f"on XY turned {TURNS}° Shift+F moves the focus XYZ by {turned}, not a voxel along its normal")

    async def choose(side, name):
        await page.click(f'[data-{side}="{name}"]')
        shown = (await page.evaluate(SHOWN))["left" if side == "ltab" else "right"]
        if shown["tab"] != name or [panel["name"] for panel in shown["panels"]] != [name]:
            failures.append(f"choosing the {name} tab shows {shown['tab']}'s tab and panels "
                            f"{[panel['name'] for panel in shown['panels']]}")
            return []
        return shown["panels"][0]

    for name, cards in LEFT_CARDS.items():
        panel = await choose("ltab", name)
        if panel and not set(cards) <= set(panel["cards"]):
            failures.append(f"the {name} tab holds {panel['cards']}, not {cards}")
    kept = await page.evaluate("document.querySelector('#scale-select').value")
    found = []
    for keys in (["End"], ["K", "K"]):
        await page.click("#scale-select")
        opened = await page.evaluate("document.querySelector('#scale-select').matches(':open')")
        await page.key("Escape")
        shown = []
        for key in keys:
            await page.key(key)
            shown.append(await page.evaluate("document.querySelector('#toggle-left').ariaExpanded"))
        found.append([opened, shown, await page.evaluate("document.querySelector('#scale-select').value")])
    if found != (want := [[True, ["true"], kept], [True, ["false", "true"], kept]]):
        failures.append(f"#scale-select opened by a click and closed by Escape, then End, and again, then K twice, "
                        f"gives [opened, the left panel shown after each key, its value] {found}, not {want}")

    await page.reveal("#zoom")
    await page.click("#zoom")
    before = await page.evaluate(f"[{ZOOM}, {ROTATION}]")
    await page.key("ArrowRight")
    await asyncio.sleep(TURNED_S)
    after = await page.evaluate(f"[{ZOOM}, {ROTATION}]")
    if after[0] != before[0] or after[1] == before[1]:
        failures.append(f"ArrowRight after a click on the zoom slider took [its value, the Rotation card's text] from "
                        f"{before} to {after}: it should turn the view and leave the slider")
    await page.key("ArrowLeft")
    await page.click("#toggle-left")
    await page.key(" ")
    pressed = await page.evaluate(f"[{STATUS}, document.querySelector('#toggle-left').ariaExpanded]")
    await page.key(" ")
    if not pressed[0].startswith("The CT alone") or pressed[1] != "false":
        failures.append(f"Space after a click on the left panel's toggle gives [the status bar, the toggle's "
                        f"aria-expanded] {pressed}, not the CT alone with the panel still hidden")
    await page.click("#toggle-left")
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    await page.reveal("#surface-flip-u")
    await page.click("#surface-flip-u")
    hidden = await page.evaluate(HIDDEN)
    await page.key("v")
    if await page.evaluate(HIDDEN) == hidden:
        failures.append("V after a click on the Surface's Flip U checkbox did not show or hide the crosshair")
    await page.click("#surface-flip-u")
    await page.key("v")
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("u")

    rtabs = (await page.evaluate(SHOWN))["rtabs"]
    if rtabs[-1] != ["info", "Info"]:
        failures.append(f"the right panel's tabs are {rtabs}, not ending in Info")
    for name, title in rtabs:
        panel = await choose("rtab", name)
        if panel and panel["height"] < panel["body"] - HEIGHT_PX:
            failures.append(f"the {title} tab's panel is {panel['height']} px of the body's {panel['body']}")
        if panel and name == "info" and panel["cards"] != INFO_CARDS:
            failures.append(f"the Info tab holds {panel['cards']}, not {INFO_CARDS}")

    left, right = "view", rtabs[1][0]
    await choose("ltab", left)
    await choose("rtab", right)
    await asyncio.sleep(PUBLISHED_S)
    t0 = time.monotonic()
    await page.open(await page.address())
    await page.settle(t0, {"panes complete": ct_key})
    shown = await page.evaluate(SHOWN)
    if [shown["left"]["tab"], shown["right"]["tab"]] != [left, right]:
        failures.append(f"the reload shows the {shown['left']['tab']} and {shown['right']['tab']} tabs, not {left} "
                        f"and {right}")

    session = await page.address_session()
    for named, want in [(rtabs[2][0], rtabs[2][0]), ("no-such-tab", rtabs[0][0])]:
        session["tabs"] = {"left": "data", "right": named}
        t0 = time.monotonic()
        await page.open_session(session)
        await page.settle(t0, {"panes complete": ct_key})
        shown = await page.evaluate(SHOWN)
        written = (await page.address_session())["tabs"]
        if [shown["left"]["tab"], shown["right"]["tab"], written] != ["data", want, {"left": "data", "right": want}]:
            failures.append(f"a link naming the data and {named} tabs shows the {shown['left']['tab']} and "
                            f"{shown['right']['tab']} tabs and writes {written}, not data and {want}")
    return failures
