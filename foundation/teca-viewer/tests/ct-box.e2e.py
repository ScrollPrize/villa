import asyncio
import time

from page import CENTER_XYZ, NEUTRAL_XY, POLL_S, TIMEOUT_S, VOLUME, ct_key

NOTE = "Level 0, 256 × 256 × 256 voxels, 16.0 MiB, 27 chunks read, about x 20511, y 13999, z 43903."
SIDE = 256
CENTER_ZYX = [float(value) for value in reversed(CENTER_XYZ.split(","))]


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    if (place := await page.place("#ct-box-section")) != "view":
        failures.append(f"its card is on {place!r}, not the View tab")
    await page.reveal("#ct-box-section")
    t0 = time.monotonic()
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.settle(t0, {"panes complete": ct_key})

    async def picture(name):
        png, pixels = await page.picture((await page.panes())["3d"])
        (out / f"{name}.png").write_bytes(png)
        return pixels["sha256"]

    async def note():
        return await page.evaluate("document.querySelector('#ct-box-note').textContent")

    async def read(t0, about):
        place = ", ".join(f"{axis} {round(value)}" for axis, value in zip("xyz", reversed(about)))
        while not (await note()).endswith(f"about {place}."):
            if time.monotonic() - t0 > TIMEOUT_S:
                raise TimeoutError(f"the cube was not read about {place} in {TIMEOUT_S} s: {await note()}")
            await asyncio.sleep(POLL_S)
        await page.settle(t0, {"panes complete": ct_key})
        return await note()

    async def focus():
        return (await page.evaluate("window.__tecaDebug.session()"))["center_ZYX"]

    ct = await picture("ct")
    t0 = time.monotonic()
    await page.click("#ct-box-show")
    if (shown := await read(t0, CENTER_ZYX)) != NOTE:
        failures.append(f"the cube's note is {shown!r}, not {NOTE!r}")
    brightest = await picture("brightest")
    if brightest == ct:
        failures.append(f"showing the cube left the 3D pane as it was, {out / 'brightest.png'}")

    x, y = await page.pane_center("3d")
    t0 = time.monotonic()
    await page.mouse("mouseMoved", x, y)
    await page.mouse("mousePressed", x, y, button="left", buttons=1, clickCount=1)
    await page.mouse("mouseReleased", x, y, button="left", buttons=0, clickCount=1)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    await page.settle(t0, {"panes complete": ct_key})
    picked = await focus()
    moved = [abs(a - b) for a, b in zip(picked, CENTER_ZYX)]
    if max(moved) < 1 or max(moved) > SIDE / 2:
        failures.append(f"a click on the cube focused {picked}, not a voxel of the cube about {CENTER_ZYX} but its centre")

    t0 = time.monotonic()
    await page.choose("#ct-box-follow", checked=True)
    followed = await read(t0, picked)
    chunks = int(followed.split(" chunks read")[0].split(", ")[-1])
    print(f"following the focus {max(moved):.0f} voxels read {chunks} chunks and drew in {time.monotonic() - t0:.2f} s")
    if chunks >= 27:
        failures.append(f"following the focus read {chunks} chunks of the 27 about it, not only those the cube did not hold")
    t0 = time.monotonic()
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    back = await read(t0, CENTER_ZYX)
    print(f"and back: {back.split(', about')[0]}, in {time.monotonic() - t0:.2f} s")

    t0 = time.monotonic()
    await page.choose("#ct-box-draw", "absorbed")
    await page.settle(t0, {"panes complete": ct_key})
    absorbed = await picture("absorbed")
    if absorbed in (ct, brightest):
        failures.append(f"absorbing drew the {'CT' if absorbed == ct else 'brightest'} picture, {out / 'absorbed.png'}")

    t0 = time.monotonic()
    await page.click("#ct-box-show")
    await page.settle(t0, {"panes complete": ct_key})
    if await picture("hidden") != ct:
        failures.append(f"hiding the cube did not give the 3D pane back, {out / 'hidden.png'}")

    t0 = time.monotonic()
    await page.open(await page.address())
    await page.settle(t0, {"panes complete": ct_key})
    drawn, follows = await page.evaluate("['#ct-box-draw', '#ct-box-follow'].map((id) => document.querySelector(id))"
                                         ".map((e) => e.type === 'checkbox' ? e.checked : e.value)")
    if (drawn, follows) != ("absorbed", True):
        failures.append(f"the reloaded page draws the cube {drawn} and follows the focus {follows}, not absorbed and true")
    return failures
