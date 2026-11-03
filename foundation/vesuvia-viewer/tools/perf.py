import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from browser import Browser, find_free_port  # noqa: E402
from page import SETTLED, hold, settle  # noqa: E402

parser = argparse.ArgumentParser(description="Frame times on the real GPU")
parser.add_argument("hash", help="viewer hash, e.g. '#ct=URL&layer=URL&focus=Z,Y,X&zoom=8'")
parser.add_argument("--seconds", type=float, default=5.0, help="length of each phase")
parser.add_argument("--swiftshader", action="store_true", help="software rendering, for a sanity run")
parser.add_argument("--profile", action="store_true", help="sample the main thread during each phase and list the top self-time functions")
args = parser.parse_args()
sys.stdout.reconfigure(line_buffering=True)

RENDERER = ("(() => { const gl = vesuvia.panes.xy.gl; const info = gl.getExtension('WEBGL_debug_renderer_info');"
            " return gl.getParameter(info.UNMASKED_RENDERER_WEBGL) })()")
RECORDER = """(() => {
  const gl = vesuvia.panes.xy.gl
  const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2')
  if (timer === null) throw new Error('GPU timer queries are unavailable: EXT_disjoint_timer_query_webgl2')
  window.__perf = { frames: [], gpu: [], queries: [], error: null }
  for (const pane of Object.values(vesuvia.panes)) {
    const draw = pane.draw.bind(pane)
    pane.draw = () => {
      if (__perf.error !== null) return draw()
      const query = gl.createQuery()
      gl.beginQuery(timer.TIME_ELAPSED_EXT, query)
      draw()
      gl.endQuery(timer.TIME_ELAPSED_EXT)
      __perf.queries.push({ name: pane.element.id.slice(5), query })
    }
  }
  ;(function loop(last) { requestAnimationFrame(t => {
    if (gl.getParameter(timer.GPU_DISJOINT_EXT)) {
      __perf.error = 'GPU timer became disjoint; timing results are invalid. Rerun the benchmark.'
      for (const { query } of __perf.queries) gl.deleteQuery(query)
      __perf.queries = []
      __perf.gpu = []
      return
    }
    if (last !== null) __perf.frames.push(t - last)
    while (__perf.queries.length > 0 && gl.getQueryParameter(__perf.queries[0].query, gl.QUERY_RESULT_AVAILABLE)) {
      const { name, query } = __perf.queries.shift()
      __perf.gpu.push([name, gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6])
      gl.deleteQuery(query)
    }
    loop(t)
  }) })(null)
})()"""
PLANNING = """(async () => {
  const { planChunks } = await import('./views/plan.js')
  const { WINDOW_CHUNKS, PIN_COARSEST_CHUNKS } = await import('./config.js')
  const scene = vesuvia.scene, panes = Object.values(vesuvia.panes), times = []
  const minDownsample = scene.ct.residency.levels[scene.levelCap].downsample_zyx[0]
  for (let i = 0; i < 21; i++) {
    const start = performance.now()
    const views = panes.map(pane => ({ boxes: pane.demand(), zoom: pane.worldZoom() }))
    for (const { residency } of [scene.ct, ...scene.layers])
      planChunks({ levels: residency.levels, views, minDownsample, slotCapacity: residency.slotCount, windowChunks: WINDOW_CHUNKS, pinCoarsestChunks: PIN_COARSEST_CHUNKS })
    if (i > 0) times.push(performance.now() - start)
  }
  return times.sort((a, b) => a - b)
})()"""
LONG_FRAME_MS = 20
WHEEL_S = 0.05
DRAG_S = 1 / 60
DRAG_RADIUS_PX = 150
ZOOM_NOTCHES = 12
SHIFT = 8


async def pane_centre(browser, name):
    rect = await browser.evaluate(f"vesuvia.panes.{name}.element.getBoundingClientRect().toJSON()")
    return rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2


async def mouse(browser, type, x, y, **params):
    await browser.send("Input.dispatchMouseEvent", {"type": type, "x": x, "y": y, **params})


async def wheel(browser, x, y, seconds, modifiers=0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await mouse(browser, "mouseWheel", x, y, deltaX=0, deltaY=100, modifiers=modifiers)
        await asyncio.sleep(WHEEL_S)


async def drag(browser, x, y, seconds, button="left"):
    await mouse(browser, "mousePressed", x, y, button=button, clickCount=1)
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        angle = (time.monotonic() - start) * math.pi
        await mouse(browser, "mouseMoved", x + DRAG_RADIUS_PX * math.sin(angle), y + DRAG_RADIUS_PX * (1 - math.cos(angle)), button=button, buttons=2 if button == "right" else 1)
        await asyncio.sleep(DRAG_S)
    await mouse(browser, "mouseReleased", x, y, button=button, clickCount=1)


async def zoom(browser, x, y, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for delta in [100] * ZOOM_NOTCHES + [-100] * ZOOM_NOTCHES:
            await mouse(browser, "mouseWheel", x, y, deltaX=0, deltaY=delta)
            await asyncio.sleep(WHEEL_S * 2)


async def task_seconds(browser):
    metrics = await browser.send("Performance.getMetrics")
    return next(m["value"] for m in metrics["metrics"] if m["name"] == "TaskDuration")


def self_times(profile):
    nodes = {node["id"]: node for node in profile["nodes"]}
    ms = {}
    for node_id, delta in zip(profile["samples"], profile["timeDeltas"]):
        ms[node_id] = ms.get(node_id, 0) + delta / 1000
    rows = []
    for node_id, total in ms.items():
        frame = nodes[node_id]["callFrame"]
        where = f"{frame['functionName'] or '(anonymous)'} {frame['url'].rsplit('/', 1)[-1]}:{frame['lineNumber'] + 1}"
        rows.append((total, where))
    return sorted(rows, reverse=True)


async def run_phase(browser, name, action):
    await browser.evaluate("__perf.frames = []; __perf.gpu = []")
    if args.profile:
        await browser.send("Profiler.start")
    busy0, wall0 = await task_seconds(browser), time.monotonic()
    await action()
    busy = (await task_seconds(browser) - busy0) / (time.monotonic() - wall0)
    profile = (await browser.send("Profiler.stop"))["profile"] if args.profile else None
    await asyncio.sleep(0.1)
    error = await browser.evaluate('__perf.error')
    if error is not None:
        raise RuntimeError(error)
    frames = sorted(await browser.evaluate("__perf.frames"))
    gpu = await browser.evaluate("__perf.gpu")
    end = await browser.evaluate("vesuvia.stats()")
    pose = await browser.evaluate("({ focus: vesuvia.scene.focus_zyx.map(Math.round), zoom: vesuvia.scene.zoom.toPrecision(3),"
                                  " inkLevels: Object.values(vesuvia.panes).map(p => p.displayLevel.layers) })")
    started = time.monotonic()
    await settle(browser, 120)
    catch_up = time.monotonic() - started

    def percentile(q):
        return frames[min(len(frames) - 1, int(q * len(frames)))]

    print(f"{name:16} {len(frames):4d} frames  interval p50 {percentile(0.5):5.1f} p95 {percentile(0.95):5.1f} max {max(frames):6.1f} ms"
          f"  >{LONG_FRAME_MS} ms: {sum(f > LONG_FRAME_MS for f in frames):3d}"
          f"  | query elapsed {sum(ms for _, ms in gpu) / len(frames):5.2f} ms/frame  | main thread {busy:4.0%}"
          f"  | end: missing {end['missing']:3d} in flight {end['inFlight']} resident {end['residentMiB']:4.0f} MiB CT levels {end['displayLevels']} ink levels {pose['inkLevels']}"
          f"  | caught up {catch_up:4.1f} s  | focus {pose['focus']} zoom {pose['zoom']}")
    per_pane = []
    for pane in ["xy", "xz", "yz", "surface"]:
        draws = sorted(ms for name, ms in gpu if name == pane)
        if draws:
            per_pane.append(f"{pane} p50 {draws[len(draws) // 2]:4.2f} p95 {draws[int(0.95 * len(draws))]:4.2f} max {draws[-1]:5.2f}")
    print(f"{'':16} query elapsed ms per draw: {'  |  '.join(per_pane)}")
    if profile is not None:
        rows = self_times(profile)
        outline = sum(ms for ms, where in rows if " outline.js:" in where)
        planner = sum(ms for ms, where in rows if " plan.js:" in where)
        print(f"{'':16} sampled self time: outline {outline:.1f} ms | planner {planner:.1f} ms")
        for ms, where in rows[:8]:
            print(f"{'':16} {ms:7.0f} ms  {where}")


async def main():
    port = find_free_port()
    server = subprocess.Popen([sys.executable, str(REPO / "tools/serve.py"), "--port", str(port), "--dir", str(REPO)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with Browser(gpu=not args.swiftshader) as browser:
            await browser.send("Network.enable")
            await browser.send("Network.setCacheDisabled", {"cacheDisabled": True})
            await browser.send("Performance.enable")
            if args.profile:
                await browser.send("Profiler.enable")
            await browser.navigate(f"http://127.0.0.1:{port}/{args.hash}")
            await browser.wait_for("window.vesuvia !== undefined", 10)
            print(f"renderer: {await browser.evaluate(RENDERER)}")
            print("Query elapsed time includes GPU waits for CPU command submission; it does not isolate GPU execution cost.")
            started = time.monotonic()
            await browser.wait_for(SETTLED, 300)
            error = await browser.evaluate("document.getElementById('error').textContent")
            if error:
                raise RuntimeError(error)
            if not await browser.evaluate("!new URLSearchParams(location.hash.slice(1)).has('surface') || vesuvia.scene.surface !== null"):
                raise RuntimeError("requested surface is missing")
            print(f"first settle: {time.monotonic() - started:.1f} s  {await browser.evaluate('vesuvia.stats()')}")
            planning = await browser.evaluate(PLANNING)
            print(f"demand + planning, all sources: p50 {planning[len(planning)//2]:.2f} p95 {planning[int(len(planning)*0.95)]:.2f} ms")
            initial_pose = await browser.evaluate("({focus_zyx: vesuvia.scene.focus_zyx, zoom: vesuvia.scene.zoom, axes_zyx: vesuvia.scene.axes_zyx})")
            await browser.evaluate(RECORDER)
            xy = await pane_centre(browser, "xy")
            has_surface = await browser.evaluate("vesuvia.scene.surface !== null")
            surface = await pane_centre(browser, "surface") if has_surface else None
            phases = [
                ("xy scroll", lambda: wheel(browser, *xy, args.seconds, modifiers=SHIFT)),
                ("xy held depth", lambda: hold(browser, "f", "KeyF", args.seconds)),
                ("xy drag", lambda: drag(browser, *xy, args.seconds)),
                ("xy zoom", lambda: zoom(browser, *xy, args.seconds)),
            ]
            if has_surface:
                phases += [
                    ("surface scroll", lambda: wheel(browser, *surface, args.seconds, modifiers=SHIFT)),
                    ("surface zoom", lambda: zoom(browser, *surface, args.seconds)),
                    ("surface drag", lambda: drag(browser, *surface, args.seconds)),
                ]
            phases.append(("xy rotate", lambda: drag(browser, *xy, args.seconds, button="right")))
            for name, action in phases:
                if name == "xy rotate":
                    await browser.evaluate(f"Object.assign(vesuvia.scene, {json.dumps(initial_pose)}); vesuvia.onChange()")
                    await asyncio.sleep(0.15)
                    await browser.wait_for(SETTLED, 120)
                await run_phase(browser, name, action)
            for line in browser.console:
                if line["level"] == "error":
                    print(f"console error: {line['text']}")
    finally:
        server.terminate()


asyncio.run(main())
