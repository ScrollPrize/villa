import time

from page import SURFACE, VOLUME, ct_key, lines_key, surface_key

VIEWS = {"xy": ["xy"], "zy": ["zy"], "xz": ["xz"], "3d": ["3d"], "grid": ["3d", "xy", "xz", "zy"]}
MARKED = "[...document.querySelectorAll('#view-strip [aria-pressed=true]')].map((button) => button.dataset.view)"
BUTTONS = """Object.fromEntries([...document.querySelectorAll('#view-strip [data-view]')].map((button) =>
  [button.dataset.view, [button.textContent.trim(), button.querySelector('svg') !== null, button.dataset.tip]]))"""
FACES = {"grid": ["", True, "Four panes", "Z"], "xy": ["XY", False, "XY", "1"], "zy": ["YZ", False, "YZ", "2"],
         "xz": ["XZ", False, "XZ", "3"], "3d": ["3D", False, "3D", "4"], "surface": ["Surface", False, "Surface", "U"]}
LOCK = """(() => {   // where Lock normal is, whether it is shown and pressed, and whether the offset is held
  const lock = document.querySelector("#surface-normal-lock");
  return [lock.closest("#top-bar") !== null, !lock.hidden, lock.getAttribute("aria-pressed"),
          document.querySelector("#surface-offset").disabled];
})()"""


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}&overlay={SURFACE}")
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    buttons = await page.evaluate(BUTTONS)
    for name, (text, icon, view, key) in FACES.items():
        got_text, got_icon, tip = buttons[name]
        if [got_text, got_icon] != [text, icon] or not tip.startswith(view) or f"({key}" not in tip:
            failures.append(f"{name}'s button reads {got_text!r} with icon {got_icon} and tip {tip!r}, not {text!r} with "
                            f"icon {icon} and a tip naming {view} and ({key}")

    async def shown(name, panes):
        got = sorted(await page.panes())
        marked = await page.evaluate(MARKED)
        if got != panes or marked != [name]:
            failures.append(f"{name} shows {got} and marks {marked}, not {panes} and {name}")

    for name, panes in VIEWS.items():
        await page.click(f'#view-strip [data-view="{name}"]')
        await shown(name, panes)
    await page.evaluate("document.activeElement.blur(), true")
    await page.key("2")
    await page.key("z", down_only=True)
    await page.key("z", repeat=True)
    await shown("zy", ["zy"])

    t0 = time.monotonic()
    await page.click('#view-strip [data-view="surface"]')
    await page.settle(t0, {"surface": surface_key})
    await shown("surface", ["surface"])
    await page.choose("#surface-layout-mode", "below")
    await page.click('#view-strip [data-view="xy"]')
    await shown("xy", ["xy"])
    await page.click('#view-strip [data-view="surface"]')
    await shown("surface", ["surface"])
    await page.click("#surface-normal-lock")
    if (lock := await page.evaluate(LOCK)) != [True, True, "true", True]:
        failures.append(f"Lock normal is in the top bar, shown, pressed and holding the offset as {lock}, not all")
    if "(Shift+L" not in (tip := await page.evaluate("document.querySelector('#surface-normal-lock').title")):
        failures.append(f"Lock normal's tip {tip!r} does not name Shift+L")
    await page.key("L", down_only=True, shift=True)
    await page.key("L", repeat=True, shift=True)
    if (lock := await page.evaluate(LOCK)) != [True, True, "false", False]:
        failures.append(f"Shift+L held until it repeats leaves Lock normal shown, pressed and the offset held as {lock}, "
                        "not unlocked once")
    await page.key("L", shift=True)
    if (lock := await page.evaluate(LOCK)) != [True, True, "true", True]:
        failures.append(f"Shift+L again leaves Lock normal shown, pressed and the offset held as {lock}, not locked")
    await page.click('#view-strip [data-view="grid"]')
    await shown("grid", VIEWS["grid"])
    if (await page.evaluate(LOCK))[1]:
        failures.append("Lock normal is shown with the Surface closed")
    await page.key("L", shift=True)
    if (lock := await page.evaluate(LOCK))[2] != "true":
        failures.append(f"Shift+L with the Surface closed leaves Lock normal pressed as {lock[2]}, not true")
    return failures
