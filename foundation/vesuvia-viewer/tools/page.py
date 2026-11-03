import asyncio
import time

SETTLED = ("window.vesuvia && !vesuvia.scene.loading && !vesuvia.scene.dirty && vesuvia.scene.ct !== null && vesuvia.stats().wanted > 0"
           " && vesuvia.stats().missing === 0 && vesuvia.stats().pending === 0")
KEY_REPEAT_S = 0.03
PIXEL_TOLERANCE = 3


async def settle(browser, timeout=60):
    # A page timer of REPLAN_MS fires after any replan the last change deferred; a sleep here can be overtaken.
    await browser.evaluate("import('./config.js').then(({ REPLAN_MS }) => new Promise(resolve => setTimeout(resolve, REPLAN_MS)))")
    await browser.wait_for(SETTLED, timeout)


def centre_pixel(name, dx=0, dy=0):
    # This callback runs after the page's own in the same frame, while the drawing buffer still holds what it drew.
    return (f"import('./gpu/gl.js').then(({{ paneRect }}) => new Promise(resolve => {{"
            f" vesuvia.scene.dirty = true; requestAnimationFrame(() => {{"
            f" const pane = vesuvia.panes.{name}, {{ gl }} = pane, [cx, cy] = pane.center(), rect = paneRect(gl.canvas, pane.rect), pixel = new Uint8Array(4);"
            f" gl.readPixels(rect.x + Math.round((Math.floor(cx) + {dx}) * devicePixelRatio), rect.y + rect.height - 1 - Math.round((Math.floor(cy) + {dy}) * devicePixelRatio),"
            f" 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel); resolve(Array.from(pixel)) }}) }}))")


async def hold(browser, key, code, seconds):
    event = {"key": key, "code": code, "windowsVirtualKeyCode": ord(key.upper()), "nativeVirtualKeyCode": ord(key.upper())}
    await browser.send("Input.dispatchKeyEvent", {"type": "keyDown", **event})
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await asyncio.sleep(KEY_REPEAT_S)
        await browser.send("Input.dispatchKeyEvent", {"type": "keyDown", "autoRepeat": True, **event})
    await browser.send("Input.dispatchKeyEvent", {"type": "keyUp", **event})


async def gl_errors(browser):
    error = await browser.evaluate("vesuvia.panes.xy.gl.getError()")
    return [] if error == 0 else [f"WebGL error {error}"]
