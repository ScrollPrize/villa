import json
import math
import re
import time

from page import CENTER_XYZ, VOLUME, ct_key

ZOOM = 1
TURNS = {"ArrowRight": 30, "ArrowUp": 20}
CLICKS_PX = [(-80, -60), (80, 60), (80, -60)]   # 120 px apart or more, so a whole voxel turns a line by under 0.01
KEPT = ", kept in this browser."
NOTE = "document.querySelector('#points-note').textContent"
SQUARE = 0.02
STORED_NML = f"""new Promise((resolve, reject) => {{
  const opening = indexedDB.open("teca-points");
  opening.onerror = () => reject(opening.error);
  opening.onsuccess = () => {{
    const read = opening.result.transaction("skeletons").objectStore("skeletons").get({json.dumps(VOLUME)});
    read.onerror = () => reject(read.error);
    read.onsuccess = () => {{
      opening.result.close();
      resolve(read.result ?? null);
    }};
  }};
}})"""
NODE = re.compile(r'<node [^>]*x="([^"]*)" y="([^"]*)" z="([^"]*)"([^>]*)/>')
VIEW = re.compile(r'teca\.pane="([^"]*)" teca\.normal_ZYX="([^"]*)"')


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.reveal("#points-section")
    await page.choose("#points-follow", checked=False)
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    await page.choose("#zoom-number", str(ZOOM))
    rect = (await page.panes())["xy"]
    x, y = round(rect["x"] + rect["width"] / 2), round(rect["y"] + rect["height"] / 2)
    await page.mouse("mouseMoved", x, y)
    for key, presses in TURNS.items():
        for _ in range(presses):
            await page.key(key)
    t0 = time.monotonic()
    await page.settle(t0, {"panes complete": ct_key})

    await page.click("#points-tool")
    for dx, dy in CLICKS_PX:
        await page.mouse("mouseMoved", x + dx, y + dy)
        await page.mouse("mousePressed", x + dx, y + dy, button="left", buttons=1, clickCount=1)
        await page.mouse("mouseReleased", x + dx, y + dy, button="left", buttons=0, clickCount=1)
        await page.wait_text(NOTE, lambda note: note.endswith(KEPT), 60)
    stored = await page.evaluate(STORED_NML)
    nodes = [] if stored is None else NODE.findall(stored)
    if len(nodes) != len(CLICKS_PX):
        failures.append(f"IndexedDB keeps {len(nodes)} nodes, not the {len(CLICKS_PX)} clicked")
        return failures
    points_ZYX = [[float(z), float(y), float(x)] for x, y, z, _ in nodes]
    for (_, _, _, rest), point_ZYX in zip(nodes, points_ZYX):
        view = VIEW.search(rest)
        if view is None or view[1] != "xy":
            failures.append(f"a node at z, y, x {point_ZYX} keeps {rest.strip()!r}, not teca.pane=\"xy\" and a normal")
            continue
        normal_ZYX = [float(value) for value in view[2].split(",")]
        if abs(math.hypot(*normal_ZYX) - 1) > SQUARE or normal_ZYX == [1, 0, 0]:
            failures.append(f"a node on the turned xy pane keeps the normal {normal_ZYX}")
            continue
        for other_ZYX in points_ZYX:
            line_ZYX = [b - a for a, b in zip(point_ZYX, other_ZYX)]
            if math.hypot(*line_ZYX) == 0:
                continue
            cos = sum(n * d for n, d in zip(normal_ZYX, line_ZYX)) / math.hypot(*line_ZYX)
            if abs(cos) > SQUARE:
                failures.append(f"the normal {normal_ZYX} is at cos {cos:.3f} to the line from {point_ZYX} to "
                                f"{other_ZYX}, which lies in the pane")
    return failures
