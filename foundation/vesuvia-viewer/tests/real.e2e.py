import json
from pathlib import Path

from page import PIXEL_TOLERANCE, SETTLED, centre_pixel, gl_errors, settle

REPO = Path(__file__).resolve().parent.parent
PROBES = json.loads((REPO / "real/probes.json").read_text())
FOCUS = PROBES["focus"]
HASH = f"#ct=real/ct.zarr&layer=real/ink.zarr&surface=real/surface.tifxyz&focus={','.join(str(f) for f in FOCUS)}&zoom=8"
LEVELS = 6


async def check(browser):
    failures = []
    await browser.wait_for(SETTLED, 120)
    registered = await browser.evaluate("""[vesuvia.scene.ct, ...vesuvia.scene.layers].map(source => ({
      downsample: source.volume.levels.map(level => level.downsample_zyx), offsets: source.volume.levels.map(level => level.offset_zyx),
      chunk: source.volume.levels[0].chunk_zyx, brick: source.residency.levels[0].chunk_zyx }))""")
    expected = [{"downsample": [[2 ** k] * 3 for k in range(LEVELS)], "offsets": [[0, 0, 0]] * LEVELS, "chunk": [chunk] * 3, "brick": [128] * 3}
                for chunk in (128, 256)]
    if registered != expected:
        failures.append(f"registered {registered}, expected {expected}")

    await browser.evaluate("vesuvia.scene.overlays = { outline: false, crosshair: false };"
                           " vesuvia.scene.layers.forEach(layer => layer.style.visible = false); vesuvia.onChange()")
    for probe in PROBES["ct"]:
        await browser.evaluate(f"vesuvia.scene.zoom = {2 ** probe['level']}; vesuvia.scene.focus_zyx = {probe['world']}; vesuvia.onChange()")
        await settle(browser)
        drawn = [await browser.evaluate(centre_pixel(name)) for name in ("xy", "xz", "yz")]
        shown = await browser.evaluate("[vesuvia.panes.xy.displayLevel.ct, vesuvia.panes.xz.displayLevel.ct, vesuvia.panes.yz.displayLevel.ct]")
        if shown != [probe["level"]] * 3 or any(abs(r - probe["value"]) > PIXEL_TOLERANCE or r != g or g != b for r, g, b, _ in drawn):
            failures.append(f"level {probe['level']} at {probe['world']}: panes drew {drawn} from levels {shown}, CT value {probe['value']}")
    read = await browser.evaluate(f"""(async () => {{
      const {{ readRegion }} = await import('./volume/region.js');
      const sources = {{ ct: vesuvia.scene.ct.volume, ink: vesuvia.scene.layers[0].volume }};
      const values = {{}};
      for (const [name, rows] of Object.entries({json.dumps({name: PROBES[name] for name in ("ct", "ink")})}))
        values[name] = await Promise.all(rows.map(async row => (await readRegion(sources[name], row.level, row.voxel, [1, 1, 1]))[0]));
      return values;
    }})()""")
    if read != {name: [row["value"] for row in PROBES[name]] for name in ("ct", "ink")}:
        failures.append(f"readRegion read {read} at the probe voxels")

    surface = PROBES["surface"]
    await browser.evaluate(f"vesuvia.scene.overlays = {{ outline: true, crosshair: false }}; vesuvia.scene.zoom = {2 ** surface['level']};"
                           f" vesuvia.scene.focus_zyx = {surface['zyx']}; vesuvia.onChange()")
    await settle(browser)
    amber = False
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            r, g, b, _ = await browser.evaluate(centre_pixel("xy", dx, dy))
            amber = amber or (r > 200 and g > 150 and b < 140)
    if not amber:
        failures.append(f"outline at {surface['zyx']}: no amber pixel within 2 px of the XY centre")
    await browser.evaluate(f"""(() => {{ const scene = vesuvia.scene, [z, y, x] = {surface['zyx']};
      scene.focus_zyx = [z + 200, y, x];
      scene.surfaceView = {{ center_uv: {surface['uv']}, zoom: {2 ** surface['level']} / scene.surface.cellSpacing, offset: 0,
        slab: {{ front: 0, back: 0, step: 1, mode: 'single' }} }};
      vesuvia.onChange() }})()""")
    await settle(browser)
    r, g, b, _ = await browser.evaluate(centre_pixel("surface"))
    if not r == g == b or r <= surface["chunkMean"]:
        failures.append(f"surface pane at uv {surface['uv']}: drew {(r, g, b)}, expected the sheet above the chunk's mean {surface['chunkMean']:.1f}")
    await browser.screenshot(REPO / "outputs/e2e-real.png")

    return failures + await gl_errors(browser)
