import base64
import json
import time

from page import SURFACE, VOLUME, ct_key, lines_key, surface_key

DISC_PX = 24
CUT_X, RULER_X = 0.3, 0.2
CUT_WIDTH_PX = 6
CUT_RGB, RULER_RGB = [0, 1, 1], [1, 1, 0]
FIRST_RGB, SECOND_RGB = [1, 0, 0], [0, 1, 0]
WIDE_PX = 48
OUT_PX = 18
REACHED_PX = [1, 2, 3]
DOT_RGB = [1, 0, 1]
GREEN_RGB, RED_RGB, BLUE_RGB = [0, 1, 0], [1, 0, 0], [0, 0, 1]
SHARED = [0.3, 0.3]
APART_PX = 40
NEAR = 60
PIXELS = """(async ([png, points]) => {
  const bitmap = await createImageBitmap(await (await fetch("data:image/png;base64," + png)).blob(),
    { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  return points.map(([x, y]) => [...context.getImageData(x, y, 1, 1).data.slice(0, 3)]);
})"""
CUT = f"""(async (baseVoxels) => {{
  const span = baseVoxels / __tecaDebug.session().zoom;
  const [z, y, x] = __viewer.focusPoint();
  const [cut, ruler] = [{CUT_X}, {RULER_X}].map((fraction) => fraction * span);
  window.__cut = __viewer.addMarks({{ diameter: {DISC_PX}, width: {CUT_WIDTH_PX} }});
  __cut.set({{ points_zyx_N3: Float32Array.of(z - 3, y, x - cut, z + 3, y, x + cut),
    colors_N3: Float32Array.from([...{json.dumps(CUT_RGB)}, ...{json.dumps(CUT_RGB)}]), edges_E2: Uint32Array.of(0, 1),
    selected: -1, depth: 1 }});
  window.__ruler = __viewer.addMarks({{ diameter: {DISC_PX} }});
  __ruler.set({{ points_zyx_N3: Float32Array.of(z, y, x + ruler), colors_N3: Float32Array.from({json.dumps(RULER_RGB)}),
    edges_E2: new Uint32Array(0), selected: -1, depth: 1 }});
  return true;
}})"""
DISCS = f"""(async () => {{
  const disc = (color_RGB) => ({{ points_zyx_N3: Float32Array.from(__viewer.focusPoint()), colors_N3: Float32Array.from(color_RGB),
    edges_E2: new Uint32Array(0), selected: -1, depth: 1 }});
  const [first, second] = [__viewer.addMarks({{ diameter: {DISC_PX} }}), __viewer.addMarks({{ diameter: {DISC_PX} }})];
  second.set(disc({json.dumps(SECOND_RGB)}));
  const frames = __tecaDebug.renderStats().orthogonalFrames;
  while (__tecaDebug.renderStats().orthogonalFrames === frames) await new Promise(requestAnimationFrame);
  first.set(disc({json.dumps(FIRST_RGB)}));
  return true;
}})()"""
DOT = f"""(() => {{
  const view = __viewer.surfaceView();
  const column = view.start_HW[1] - {OUT_PX} * view.pixel * view.scale_UV[0];
  __viewer.addSurfaceMarks({{ diameter: 0, width: {WIDE_PX} }}).set({{ surface: view.surface,
    cells_N2: Float32Array.of((view.start_HW[0] + view.stop_HW[0]) / 2, column), colors_N3: Float32Array.from({json.dumps(DOT_RGB)}),
    edges_E2: Uint32Array.of(0, 0), selected: -1 }});
  return true;
}})()"""

STACKED = f"""(() => {{
  const view = __viewer.surfaceView();
  const cells = view.pixel * view.scale_UV[0];
  const at = (right, down) => [0, 1].map((axis) => view.start_HW[axis] + {json.dumps(SHARED[::-1])}[axis] *
    (view.stop_HW[axis] - view.start_HW[axis]) + [down, right][axis] * cells);
  const discs = (places, colors) => ({{ surface: view.surface, cells_N2: Float32Array.from(places.flat()),
    colors_N3: Float32Array.from(colors.flat()), edges_E2: new Uint32Array(0), selected: -1 }});
  const [first, second] = [__viewer.addSurfaceMarks({{ diameter: {DISC_PX} }}), __viewer.addSurfaceMarks({{ diameter: {DISC_PX} }})];
  first.set(discs([at({APART_PX}, 0), at(0, 0)], [{json.dumps(GREEN_RGB)}, {json.dumps(RED_RGB)}]));
  second.set(discs([at(0, 0), at(0, {APART_PX})], [{json.dumps(GREEN_RGB)}, {json.dumps(BLUE_RGB)}]));
  window.__second = second;
  return true;
}})()"""


def byte(color_RGB):
    return [round(255 * channel) for channel in color_RGB]


def near(rgb, color_RGB):
    return all(abs(value - channel) < NEAR for value, channel in zip(rgb, byte(color_RGB)))


async def check(page, out):
    failures = []

    async def pixels(pane, name, places):
        rect = (await page.panes())[pane]
        png, _ = await page.picture(rect)
        (out / f"{name}.png").write_bytes(png)
        return await page.evaluate(f"{PIXELS}({json.dumps([base64.b64encode(png).decode(), places(rect)])})")

    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}&overlay={SURFACE}")
    await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key})
    await page.extension_viewer()
    await page.evaluate(f"{CUT}({await page.config('BASE_VIEW_WIDTH_VOXELS')})")
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    row = await pixels("xy", "cut", lambda rect: [[x, rect["height"] // 2] for x in range(round(rect["width"]))])
    middle = len(row) // 2
    ruler = [x for x, rgb in enumerate(row) if near(rgb, RULER_RGB)]
    cut = [x for x, rgb in enumerate(row) if near(rgb, CUT_RGB)]
    if not ruler or not cut:
        failures.append(f"XY's middle row holds {len(cut)} pixels of the cut edge and {len(ruler)} of the ruler's disc, "
                        f"{out / 'cut.png'}")
    else:
        voxels = (sum(ruler) / len(ruler) - middle) / RULER_X
        drawn = [((x - middle) + side * CUT_WIDTH_PX / 2) / voxels for x, side in [(cut[0], 1), (cut[-1], -1)]]
        if not all(abs(end - side * CUT_X / 3) < 0.01 for end, side in zip(drawn, [-1, 1])) or len(cut) != cut[-1] - cut[0] + 1:
            failures.append(f"the edge from 3 below XY's plane to 3 above is drawn from {drawn[0]:.3f} to {drawn[1]:.3f} "
                            f"of the pane along x, not over its middle third, {-CUT_X / 3:.3f} to {CUT_X / 3:.3f}, "
                            f"{out / 'cut.png'}")
    await page.evaluate("__cut.show(Uint8Array.of(1, 0)), true")
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    [hidden] = await pixels("xy", "cut-hidden", lambda rect: [[rect["width"] // 2, rect["height"] // 2]])
    if near(hidden, CUT_RGB):
        failures.append(f"the edge is still drawn at the focus once show() hides its second end, {out / 'cut-hidden.png'}")
    await page.evaluate("__cut.remove(), __ruler.remove(), true")
    await page.evaluate(DISCS)
    await page.settle(time.monotonic(), {"panes complete": ct_key})
    [focus] = await pixels("xy", "discs", lambda rect: [[rect["width"] // 2, rect["height"] // 2]])
    if not near(focus, SECOND_RGB):
        failures.append(f"the discs at the focus on XY read {focus}, not the second handle's {byte(SECOND_RGB)} over the "
                        f"first's, {out / 'discs.png'}")

    await page.key("u")
    await page.settle(time.monotonic(), {"surface": surface_key})
    await page.evaluate(DOT)
    await page.settle(time.monotonic(), {"surface": surface_key})
    reached = await pixels("surface", "dot", lambda rect: [[x, rect["height"] // 2] for x in REACHED_PX])
    if not all(near(rgb, DOT_RGB) for rgb in reached):
        failures.append(f"the Surface pane's columns {REACHED_PX} at its middle row read {reached}, not {byte(DOT_RGB)} of a "
                        f"dot {WIDE_PX} px across {OUT_PX} px past its edge, {out / 'dot.png'}")
    await page.evaluate(STACKED)
    await page.settle(time.monotonic(), {"surface": surface_key})
    [shared] = await pixels("surface", "stacked", lambda rect: [[round(rect["width"] * SHARED[0]),
                                                                 round(rect["height"] * SHARED[1])]])
    if not near(shared, GREEN_RGB):
        failures.append(f"where both Surface handles draw a disc the pane reads {shared}, not the second's "
                        f"{byte(GREEN_RGB)} over the first's {byte(RED_RGB)}, {out / 'stacked.png'}")
    await page.evaluate("__second.show(Uint8Array.of(0, 1)), true")
    await page.settle(time.monotonic(), {"surface": surface_key})
    [shown] = await pixels("surface", "shown", lambda rect: [[round(rect["width"] * SHARED[0]),
                                                              round(rect["height"] * SHARED[1])]])
    if not near(shown, RED_RGB):
        failures.append(f"once the second Surface handle's show() hides its green disc the pane reads {shown} there, not "
                        f"the first's {byte(RED_RGB)}, {out / 'shown.png'}")
    return failures
