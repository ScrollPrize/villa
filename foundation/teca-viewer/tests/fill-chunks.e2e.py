import json
import time

from page import VOLUME, ct_key

# x, y, z: level 5's chunk 0/0/0, 4,096 voxels a side, and every finer chunk inside it are 404 on the host.
CORNER = "2048, 2048, 2048"
ZOOM = "1"
RECORD = "#performance-controls [data-record]"
LEVELS = """Object.entries(window.__tecaDebug.slicedPlaneStats().panes).filter(([, pane]) => pane.visible)
  .map(([name, pane]) => [name, pane.levels.map((level) => [level.scaleIndex, level.planned, level.fills])])"""


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    t0 = time.monotonic()
    await page.choose("#zoom-number", ZOOM)
    await page.settle(t0, {"panes complete": ct_key})
    await page.reveal(RECORD)
    await page.click(RECORD)
    await page.type_into("position-number", CORNER)
    t0 = time.monotonic()
    await page.key("Enter")
    await page.settle(t0, {"panes complete": ct_key})
    await page.click(RECORD)
    report = json.loads((await page.download("#performance-controls [data-export]", "viewer-performance.json")).read_text())
    uploads = report["metrics"]["uploads"]
    if uploads["bytes"] != 0:
        failures.append(f"the jump to {CORNER} uploaded {uploads['bytes']} bytes in {uploads['jobs']} uploads, not none")
    panes = await page.evaluate(LEVELS)
    unfilled = [f"{name}'s level {scale} {planned - fills} of {planned}"
                for name, levels in panes for scale, planned, fills in levels if fills != planned]
    if unfilled:
        failures.append(f"at {CORNER} the chunks planned are not all fills: {', '.join(unfilled)} are not")
    (out / "counts.json").write_text(json.dumps({"panes": panes, "uploads": uploads}))
    return failures
