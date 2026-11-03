import asyncio
import base64
import json
import time

from page import CENTER_XYZ, NEUTRAL_XY, TIMEOUT_S, VOLUME, ct_key, lines_key, slices_key

INK = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/segments/20231221180251/"
       "ink-detection/downsampled/PHercParis4-20231221180251-2.4um-0.22m-78keV-volume-20260411134726-20260417190342-"
       "new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg")
PER_VOXEL = 3
BELOW = 0.55
# The surface's normal at CENTER_XYZ, ZYX [0.151, -0.371, 0.916], has this share in the XY plane, so the band reaches
# steps over it from the line there.
IN_PLANE = 0.988
SLACK, BEYOND = 1.5, 0.01
SHORT = 0.5
FARTHER = 0.5
# Of the line's pixels, at least this share have a pixel of the band within 1.5 voxels: the image is 0 at 9% of the
# segment's pixels, and a value under 6 changes no pixel by a grey level.
COVERED = 0.8
DRAWN_3D = 2000
OVER_MESH = 0.95
WIDE_VOXELS = 1800
EDGE = 400
EDGE_REACH = 3
WIDE_COVERED = 0.6
# The held slice's step, finest voxels a frame, how far it goes, and how far before the pane is looked at: past the
# 641 from CENTER_XYZ to the end of the second chunk of level 1 beyond the focus's, so the looks are in chunks three
# or more beyond it, which hold the band only if it is made as the moving slice reads them.
HELD_STEP, HELD_VOXELS, HELD_FROM = 8, 1200, 700
# Of the looks at the held slice, at least this share find the XY pane's level holding some of the band's chunks: at 8
# voxels a frame it holds none for up to 0.2 s at a time, as the slice moves into chunks not made yet.
HELD_SHARE = 0.5
LAYERS = "window.__tecaDebug.layerStats().length"
HOLDING = """(() => { const d = window.__tecaDebug; const pane = d.layerStats()[0].panes.xy;
  const level = pane.levels.find((level) => level.scaleIndex === pane.scaleIndex);
  return [d.session().center_ZYX[0], level.available - level.fills]; })()"""
FILLS = """Object.values(window.__tecaDebug.layerStats()[0].panes).filter((pane) => pane.visible)
  .flatMap((pane) => pane.levels).reduce(([planned, fills], level) => [planned + level.planned, fills + level.fills], [0, 0])"""
EDGE_BAND = """(async ([before, after, beyond, reach]) => {
  const rgba = async (text) => {
    const blob = await (await fetch("data:image/png;base64," + text)).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    return { width: bitmap.width, height: bitmap.height, data: context.getImageData(0, 0, bitmap.width, bitmap.height).data };
  };
  const [b, a] = await Promise.all([rgba(before), rgba(after)]);
  const band = (x, y) => {
    const i = 4 * (y * b.width + x);
    return a.data[i + 1] - b.data[i + 1] - (a.data[i] - b.data[i]) >= 6;
  };
  let line = 0, covered = 0;
  for (let y = reach; y < b.height - reach; y++)
    for (let x = reach; x < b.width - reach; x++) {
      const i = 4 * (y * b.width + x);
      if (Math.max(Math.abs(x - b.width / 2), Math.abs(y - b.height / 2)) <= beyond || b.data[i] - b.data[i + 2] < 80) continue;
      line++;
      search: for (let dy = -reach; dy <= reach; dy++)
        for (let dx = -reach; dx <= reach; dx++)
          if (band(x + dx, y + dy)) {
            covered++;
            break search;
          }
    }
  return { line, covered };
})"""
MESH = """(async ([url, reach]) => {
  const THREE = await import("./vendor/three.module.js");
  const { grid_HW, points_zyx_HW3 } = await __viewer.readSurface(url, new AbortController().signal);
  const focus_ZYX = __viewer.focusPoint();
  const zyx = (row, column) => points_zyx_HW3.subarray(3 * (row * grid_HW[1] + column), 3 * (row * grid_HW[1] + column + 1));
  const near = (point_ZYX) => point_ZYX.every(Number.isFinite) && point_ZYX.some((value) => value !== -1) &&
    point_ZYX.every((value, axis) => Math.abs(value - focus_ZYX[axis]) <= reach);
  const positions = [];
  for (let row = 0; row + 2 < grid_HW[0]; row += 2)
    for (let column = 0; column + 2 < grid_HW[1]; column += 2) {
      const corners = [zyx(row, column), zyx(row, column + 2), zyx(row + 2, column), zyx(row + 2, column + 2)];
      if (!corners.every(near)) continue;
      const [first, next, below, last] = corners.map((point_ZYX) => [...point_ZYX].reverse());
      positions.push(...first, ...below, ...next, ...next, ...below, ...last);
    }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(positions), 3));
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshNormalMaterial({ side: THREE.DoubleSide }));
  window.__mesh = { mesh, handle: __viewer.add3d(mesh) };
  return positions.length / 9;
})"""
UNMESH = "(() => { __mesh.handle.remove(); __mesh.mesh.geometry.dispose(); __mesh.mesh.material.dispose(); return true })()"
MESH_REACH = 256
GPU = "(() => { const r = window.__tecaDebug.rendererStats()['3d']; return [r.textures, r.geometries] })()"
BAND = """(async ([before, after, perVoxel, below]) => {
  const rgba = async (text) => {
    const blob = await (await fetch("data:image/png;base64," + text)).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    return { width: bitmap.width, height: bitmap.height, data: context.getImageData(0, 0, bitmap.width, bitmap.height).data };
  };
  const [b, a] = await Promise.all([rgba(before), rgba(after)]);
  const line = [], band = [];
  const inBand = new Uint8Array(b.width * b.height);
  for (let y = Math.round(below * b.height); y < b.height; y++)
    for (let x = 0; x < b.width; x++) {
      const i = 4 * (y * b.width + x);
      if (b.data[i] - b.data[i + 2] >= 80) line.push([x, y]);
      else if (a.data[i + 1] - b.data[i + 1] - (a.data[i] - b.data[i]) >= 6) {
        band.push([x, y]);
        inBand[y * b.width + x] = 1;
      }
    }
  const distances = band.map(([x, y]) => {
    let nearest = Infinity;
    for (const [lx, ly] of line) nearest = Math.min(nearest, (x - lx) ** 2 + (y - ly) ** 2);
    return Math.sqrt(nearest) / perVoxel;
  }).sort((p, q) => p - q);
  const reach = Math.floor(1.5 * perVoxel);
  const covered = line.filter(([x, y]) => {
    for (let dy = -reach; dy <= reach; dy++)
      for (let dx = -reach; dx <= reach; dx++)
        if (dx * dx + dy * dy <= reach * reach && inBand[(y + dy) * b.width + x + dx] === 1) return true;
    return false;
  }).length;
  return { line: line.length, covered, distances };
})"""
THREE_D = """(async ([nothing, ink, mesh, both]) => {
  const rgba = async (text) => {
    const blob = await (await fetch("data:image/png;base64," + text)).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    context.drawImage(bitmap, 0, 0);
    return context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  };
  const [n, i, m, b] = await Promise.all([nothing, ink, mesh, both].map(rgba));
  const differs = (p, q, at) => p[at] !== q[at] || p[at + 1] !== q[at + 1] || p[at + 2] !== q[at + 2];
  const greener = (p, q, at) => q[at + 1] - p[at + 1] > Math.max(q[at] - p[at], q[at + 2] - p[at + 2]);
  const counts = { ink: 0, inkGreener: 0, both: 0, over: 0 };
  for (let at = 0; at < n.length; at += 4) {
    if (!differs(n, i, at)) continue;
    counts.ink++;
    if (greener(n, i, at)) counts.inkGreener++;
    if (!differs(n, m, at)) continue;
    counts.both++;
    if (differs(m, b, at) && greener(m, b, at)) counts.over++;
  }
  return counts;
})"""


def encoded(png):
    return base64.b64encode(png).decode()


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.type_into("position-number", CENTER_XYZ)
    await page.key("Enter")
    t0 = time.monotonic()
    await page.add_surface()
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    rects = await page.panes()
    base_voxels = await page.config("BASE_VIEW_WIDTH_VOXELS")
    session = await page.evaluate("window.__tecaDebug.session()")
    session["zoom"] = PER_VOXEL * base_voxels / min(rects["xy"]["width"], rects["xy"]["height"])
    session["extensions"]["ink"] = [{"location": INK, "tiff": None, "surface": session["overlays"][0]["path"],
                                     "color": "#00ff00", "opacity": 1, "visible": True, "volume": False, "steps": 4,
                                     "range": [0, 255]}]

    async def settled(name, panes, band=True):
        keys = {"panes complete": ct_key, "lines settled": lines_key}
        if band:
            await page.wait_text(LAYERS, lambda layers: layers == 1, TIMEOUT_S)
            keys["band complete"] = lambda s: slices_key(s["layers"][0])
        await page.settle(time.monotonic(), keys)
        pictures = {}
        for pane in panes:
            png, pixels = await page.picture(rects[pane])
            (out / f"{name}-{pane}.png").write_bytes(png)
            pictures[pane] = (png, pixels["sha256"])
        return pictures

    async def band(name, before, after, steps):
        result = await page.evaluate(f"{BAND}({json.dumps([encoded(before), encoded(after), PER_VOXEL, BELOW])})")
        distances = result["distances"]
        if not distances or result["line"] == 0:
            failures.append(f"{name}: the pane shows {len(distances)} pixels of the band and {result['line']} of the line")
            return None
        reach = steps / IN_PLANE
        beyond = sum(distance > reach + SLACK for distance in distances) / len(distances)
        farthest = distances[int(0.98 * len(distances))]
        covered = result["covered"] / result["line"]
        print(f"{name}: {len(distances)} pixels of the band over {result['line']} of the line, {covered:.3f} of it "
              f"covered; half within {distances[len(distances) // 2]:.2f} voxels, 98% within {farthest:.2f}, "
              f"{beyond:.4f} past {reach + SLACK:.2f}")
        if beyond > BEYOND:
            failures.append(f"{name}: {beyond:.4f} of the band's pixels lie past {reach + SLACK:.2f} voxels from the "
                            f"line, not at most {BEYOND}, {out / f'{name}-xy.png'}")
        if not steps - SHORT <= farthest <= reach + SLACK:
            failures.append(f"{name}: the farthest 2% of the band's pixels lie from {farthest:.2f} voxels of the line, "
                            f"not {steps - SHORT} to {reach + SLACK:.2f}, {out / f'{name}-xy.png'}")
        if covered < COVERED:
            failures.append(f"{name}: {covered:.3f} of the line has the band within 1.5 voxels, not {COVERED}")
        return farthest

    t0 = time.monotonic()
    await page.open_session(session)
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.browser.wait_for("!document.querySelector('#ink-items .layer-row-name').textContent.endsWith('…')",
                                TIMEOUT_S)
    before = await settled("before", ["xy"], band=False)
    t0 = time.monotonic()
    await page.reveal("#ink-section")
    await page.choose("#ink-items .layer-row:nth-child(2) input[type=checkbox]", checked=True)
    four = await settled("four", ["xy"])
    print(f"the band at 4 steps is drawn {time.monotonic() - t0:.1f} s after it was asked for")
    four_reach = await band("four", before["xy"][0], four["xy"][0], 4)
    planned, fills = await page.evaluate(FILLS)
    print(f"the band's layer plans {planned} chunks, {fills} of them fills")
    if not 0 < fills < planned:
        failures.append(f"the band's layer plans {planned} chunks, {fills} of them fills, not some and not all")
    await page.choose("#ink-items .ink-steps", "8")
    eight = await settled("eight", ["xy"])
    eight_reach = await band("eight", before["xy"][0], eight["xy"][0], 8)
    if None not in (four_reach, eight_reach) and abs(eight_reach - four_reach - 4 / IN_PLANE) > FARTHER:
        failures.append(f"the band at 8 steps reaches {eight_reach - four_reach:.2f} voxels farther from the line than "
                        f"at 4, not {4 / IN_PLANE:.2f} give or take {FARTHER}")

    t0 = time.monotonic()
    await page.open(await page.address())
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    kept = await page.evaluate("[document.querySelector('#ink-items .ink-steps').value, "
                               "document.querySelector('#ink-items .layer-row:nth-child(2) input').checked]")
    if kept != ["8", True]:
        failures.append(f"the reloaded page's steps and In the volume are {kept}, not ['8', True]")
    reloaded = await settled("reloaded", ["xy", "3d"])
    if reloaded["xy"][1] != eight["xy"][1]:
        failures.append(f"the reloaded page drew another band, {out / 'reloaded-xy.png'}")
    gpu_ink = await page.evaluate(GPU)
    await page.extension_viewer()
    triangles = await page.evaluate(f"{MESH}({json.dumps([session['overlays'][0]['path'], MESH_REACH])})")
    print(f"the mesh in the 3D pane has {triangles} triangles")
    both = await settled("both", ["3d"])
    gpu_both = await page.evaluate(GPU)
    await page.reveal("#ink-section")
    await page.choose("#ink-items .layer-row:nth-child(2) input[type=checkbox]", checked=False)
    mesh = await settled("mesh", ["xy", "3d"], band=False)
    gpu_mesh = await page.evaluate(GPU)
    if mesh["xy"][1] != before["xy"][1]:
        failures.append(f"taking the band away did not give the XY pane back, {out / 'mesh-xy.png'}")
    await page.evaluate(UNMESH)
    nothing = await settled("nothing", ["3d"], band=False)
    gpu_nothing = await page.evaluate(GPU)
    print(f"the 3D pane's textures and geometries: {gpu_ink} with the band, {gpu_both} with the mesh as well, "
          f"{gpu_mesh} with the mesh alone and {gpu_nothing} with neither")
    took = [ink - none for ink, none in zip(gpu_ink, gpu_nothing)]
    if [both - mesh for both, mesh in zip(gpu_both, gpu_mesh)] != took:
        failures.append(f"taking the band away from the mesh left the 3D pane {gpu_mesh} textures and geometries, "
                        f"not {took} fewer than {gpu_both}, which the band took alone")
    pictures = [encoded(shown["3d"][0]) for shown in (nothing, reloaded, mesh, both)]
    counts = await page.evaluate(f"{THREE_D}({json.dumps(pictures)})")
    print(f"the 3D pane: the ink alone changes {counts['ink']} pixels, {counts['inkGreener']} greener; of the "
          f"{counts['both']} the mesh alone changes too, the ink over the mesh changes {counts['over']} greener")
    if counts["ink"] < DRAWN_3D or counts["inkGreener"] < 0.95 * counts["ink"]:
        failures.append(f"the ink alone changes {counts['ink']} of the 3D pane's pixels, {counts['inkGreener']} "
                        f"greener, not {DRAWN_3D}, 95% greener, {out / 'reloaded-3d.png'}")
    if counts["both"] == 0 or counts["over"] < OVER_MESH * counts["both"]:
        failures.append(f"over the mesh the ink changes {counts['over']} of the {counts['both']} pixels the ink and "
                        f"the mesh each change alone, not {OVER_MESH} of them, {out / 'both-3d.png'}")

    session.update(fullscreenViewport="xy", activeViewport="xy", zoom=base_voxels / WIDE_VOXELS,
                   heldSliceStep=HELD_STEP)
    session["extensions"]["ink"][0].update(volume=False, steps=4)
    t0 = time.monotonic()
    await page.open_session(session)
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.browser.wait_for("!document.querySelector('#ink-items .layer-row-name').textContent.endsWith('…')",
                                TIMEOUT_S)
    rects = await page.panes()
    wide_before = await settled("wide-before", ["xy"], band=False)
    await page.reveal("#ink-section")
    await page.choose("#ink-items .layer-row:nth-child(2) input[type=checkbox]", checked=True)
    wide = await settled("wide", ["xy"])
    level = (await page.evaluate("window.__tecaDebug.slicedPlaneStats().panes.xy.scaleIndex"))
    per_voxel = min(rects["xy"]["width"], rects["xy"]["height"]) / (WIDE_VOXELS / 2**level)
    edge = await page.evaluate(f"{EDGE_BAND}({json.dumps([encoded(wide_before['xy'][0]), encoded(wide['xy'][0]), EDGE * per_voxel, EDGE_REACH])})")
    print(f"the maximised pane shows level {level} at {per_voxel:.2f} CSS pixels a voxel; past {EDGE} of its voxels "
          f"from the focus, {edge['covered']} of the line's {edge['line']} pixels have the band within {EDGE_REACH}")
    if edge["line"] == 0 or edge["covered"] < WIDE_COVERED * edge["line"]:
        failures.append(f"past {EDGE} voxels of level {level} from the focus, {edge['covered']} of the line's "
                        f"{edge['line']} pixels have the band within {EDGE_REACH} CSS pixels, not {WIDE_COVERED} of "
                        f"them, {out / 'wide-xy.png'}")

    await page.mouse("mouseMoved", *await page.pane_center("xy"))
    start_z = (await page.evaluate(HOLDING))[0]
    t0 = time.monotonic()
    await page.key("f", down_only=True)
    looked = []
    while abs((holding := await page.evaluate(HOLDING))[0] - start_z) < HELD_VOXELS:
        if time.monotonic() - t0 > TIMEOUT_S:
            raise TimeoutError(f"a held slice moved {holding[0] - start_z} voxels in {TIMEOUT_S} s")
        if abs(holding[0] - start_z) >= HELD_FROM:
            looked.append(holding[1])
        await page.key("f", repeat=True, down_only=True)
        await asyncio.sleep(0.03)
    await page.browser.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "f", "code": "KeyF", "windowsVirtualKeyCode": 70})
    (out / "held-xy.png").write_bytes(await page.browser.screenshot(rects["xy"]))
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    held = sum(chunks > 0 for chunks in looked)
    print(f"from {HELD_FROM} to {HELD_VOXELS} voxels into a held slice, the XY pane's level holds the band's chunks at "
          f"{held} of {len(looked)} looks: {looked}")
    if not looked or held < HELD_SHARE * len(looked):
        failures.append(f"from {HELD_FROM} to {HELD_VOXELS} voxels into a held slice, the XY pane's level holds the "
                        f"band's chunks at {held} of {len(looked)} looks, not {HELD_SHARE} of them, {out / 'held-xy.png'}")
    return failures
