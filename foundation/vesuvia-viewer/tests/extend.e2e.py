import json
from pathlib import Path

from page import SETTLED, centre_pixel, gl_errors, settle

REPO = Path(__file__).resolve().parent.parent
FOCUS = json.loads((REPO / "real/probes.json").read_text())["focus"]
HASH = f"#ct=real/ct.zarr&surface=real/surface.tifxyz&focus={','.join(str(f) for f in FOCUS)}&zoom=8"
VOXELS = 64
SETUP = """(async () => {
  const { createMemoryVolume } = await import('./volume/memory.js')
  const { scene } = vesuvia, seen = window.seen = { slices: 0, drawn: 0, sliceDown: null, down: null, up: null }
  const offset_zyx = scene.focus_zyx.map(v => Math.floor(v) - VOXELS / 2)
  const volume = window.painted = createMemoryVolume({ name: 'painted', dtype: 'uint8', fill: 0, labels: false,
    levels: [{ shape_zyx: [VOXELS, VOXELS, VOXELS], chunk_zyx: [VOXELS, VOXELS, VOXELS], downsample_zyx: [1, 1, 1], offset_zyx }] })
  volume.write(0, [0, 0, 0], [VOXELS, VOXELS, VOXELS], new Uint8Array(VOXELS ** 3).fill(255))
  window.removeLayer = vesuvia.addLayer(volume, { style: { visible: true, opacity: 1, color: [1, 0, 0], threshold: 0, mode: 'probability' } })
  window.removeExtension = vesuvia.extend(() => ({
    panel: Object.assign(document.createElement('section'), { id: 'extend-panel' }),
    draw(pane) {
      const [z, y, x] = scene.focus_zyx
      seen.slices++
      pane.drawSegments(Float32Array.of(z, y + 200, x + 200, z, y + 300, x + 300), [1, 0, 0, 1], 1)
    },
    drawSurface(pane) { seen.drawn++; pane.drawPoints(Float32Array.of(0, 0), [1, 0, 0, 1], 4) },
    dispose() { scene.tool = scene.surfaceTool = null },
  }))
  scene.tool = { cursor: 'crosshair', status: 'Extend', down(pane, world_zyx) { seen.sliceDown = world_zyx }, move() {}, up() {} }
  scene.surfaceTool = { cursor: 'crosshair', status: 'Extend', down(pane, uv) { seen.down = uv }, move() {}, up(pane, uv) { seen.up = uv } }
  vesuvia.onChange()
})()""".replace("VOXELS", str(VOXELS))


async def check(browser):
    failures = []
    await browser.wait_for(SETTLED, 120)
    await browser.evaluate("vesuvia.scene.overlays = { outline: false, crosshair: false }; vesuvia.onChange()")
    await settle(browser)
    before = await browser.evaluate(centre_pixel("xy"))
    await browser.evaluate(SETUP)
    await settle(browser)
    painted = await browser.evaluate(centre_pixel("xy"))
    if painted != [255, 0, 0, 255]:
        failures.append(f"memory layer of red over {before} drew {painted}")
    await browser.evaluate(f"painted.write(0, [0, 0, 0], [{VOXELS}, {VOXELS}, {VOXELS}], new Uint8Array({VOXELS ** 3})); vesuvia.onChange()")
    await settle(browser)
    cleared = await browser.evaluate(centre_pixel("xy"))
    if cleared != before:
        failures.append(f"a write of zeros left {cleared}, expected the CT's {before}")
    expected = await browser.evaluate(f"""(async () => {{
      const {{ labelColor }} = await import('./gpu/layer.glsl.js')
      painted.write(0, [0, 0, 0], [{VOXELS}, {VOXELS}, {VOXELS}], new Uint8Array({VOXELS ** 3}).fill(1))
      vesuvia.scene.layers[0].style.mode = 'labels'
      vesuvia.onChange()
      return [...labelColor(1).map(c => Math.round(c * 255)), 255]
    }})()""")
    await settle(browser)
    labelled = await browser.evaluate(centre_pixel("xy"))
    if any(abs(drawn - wanted) > 1 for drawn, wanted in zip(labelled, expected)):
        failures.append(f"label 1 drew {labelled}, labelColor() says {expected}")

    for name in ("xy", "surface"):
        x, y = await browser.evaluate(f"(() => {{ const pane = vesuvia.panes.{name}, [x, y] = pane.center(); return [pane.rect.left + x, pane.rect.top + y] }})()")
        for kind in ("mousePressed", "mouseReleased"):
            await browser.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": "left", "buttons": 1, "clickCount": 1})
        await settle(browser)
    seen = await browser.evaluate("({ ...seen, expected: vesuvia.panes.surface.pixelToUv(...vesuvia.panes.surface.center()), sliceExpected: vesuvia.panes.xy.pixelToWorld(...vesuvia.panes.xy.center()), panel: document.getElementById('extend-panel') !== null })")
    if seen["slices"] == 0 or seen["drawn"] == 0 or not seen["panel"]:
        failures.append(f"draw, drawSurface or the panel did not run: {seen}")
    if seen["sliceDown"] != seen["sliceExpected"]:
        failures.append(f"scene.tool got {seen}")
    if seen["down"] != seen["expected"] or seen["up"] != seen["expected"]:
        failures.append(f"surfaceTool got {seen}")

    gone = await browser.evaluate("removeLayer(); removeExtension(); ({ layers: vesuvia.scene.layers.length, tool: vesuvia.scene.surfaceTool, panel: document.getElementById('extend-panel') })")
    if gone != {"layers": 0, "tool": None, "panel": None}:
        failures.append(f"remove left {gone}")
    return failures + await gl_errors(browser)
