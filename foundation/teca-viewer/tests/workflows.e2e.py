import asyncio
import json
import time

from page import BLANK_COLORS, CENTER_XYZ, NEUTRAL_XY, VOLUME, VOLUME_PATH, ct_key, lines_key, surface_key

ZOOM = "1"
HOLD_S = 8.0
SAMPLE_S = 0.5
KEY_REPEAT_S = 0.033
WHEEL_IN_STEPS, WHEEL_OUT_STEPS = 4, 8
WHEEL_GAP_S = 0.05
DRAG_PX, DRAG_MOVES = 300, 30
FRAME_S = 1 / 60
ROTATE_PRESSES = 10


def display_share(s):
    share = {}
    for name, scale, _, levels in s["ct"]["panes"]:
        [(available, planned)] = [(a, p) for level, a, p in levels if level == scale]
        share[name] = round(available / planned, 3) if planned > 0 else None
    return share


def traffic(browser, t0, t1):
    mine = [r for r in browser.requests.values() if t0 <= r["start"] <= t1]
    return {"requests": len(mine), "MiB": round(sum(r["bytes"] for r in mine) / 2**20, 1)}


def chunk_reads(browser, t0, t1):
    reads = [r for r in browser.requests.values() if t0 <= r["start"] <= t1 and r["method"] == "GET" and VOLUME_PATH in r["url"]
             and r["end"] is not None and r["end"] <= t1 and r["status"] in (200, 206)]
    return len(reads), round(sum(r["bytes"] for r in reads) / 2**20, 1)


async def check(page, out):
    (out / "png").mkdir(exist_ok=True)
    results = {"center_xyz": CENTER_XYZ, "timings": {}, "measures": {}, "pixels": {}, "steps": []}
    failures = []

    async def record(step, t0, times, firsts=None):
        t1 = time.monotonic()
        for name, seconds in {**times, **(firsts or {})}.items():
            results["timings"][f"{step}: {name}"] = seconds
        panes = {}
        for pane, rect in (await page.panes()).items():
            png, pixels = await page.picture(rect)
            path = out / "png" / f"{step.replace(' ', '-')}-{pane}.png"
            path.write_bytes(png)
            panes[pane] = {"png": str(path.relative_to(out)), **pixels}
            results["pixels"][f"{step}/{pane}"] = pixels["sha256"]
        position = await page.evaluate("document.querySelector('#position-number').value")
        network = traffic(page.browser, t0, t1)
        results["steps"].append({"step": step, "times": times, "firsts": firsts or {}, "position": position, "network": network, "panes": panes})
        print(f"{step:18s} " + "  ".join(f"{n} {s:.2f} s" for n, s in {**times, **(firsts or {})}.items())
              + f" | {network['requests']} requests {network['MiB']} MiB | " + " ".join(f"{p} {v['sha256'][:8]}" for p, v in panes.items()), flush=True)
        return panes

    await page.browser.send("Performance.enable")
    await page.open("")
    results["renderer"] = await page.evaluate("(() => { const g = document.createElement('canvas').getContext('webgl2'); "
                                              "return g.getParameter(g.getExtension('WEBGL_debug_renderer_info').UNMASKED_RENDERER_WEBGL) })()")

    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    times, _ = await page.settle(t0, {"panes complete": ct_key})
    await record("cold open", t0, times)

    await page.type_into("position-number", CENTER_XYZ)
    t0 = time.monotonic()
    await page.key("Enter")
    await page.choose("#zoom-number", ZOOM)
    times, _ = await page.settle(t0, {"panes complete": ct_key})
    await record("center", t0, times)

    t0 = time.monotonic()
    await page.add_surface()
    times, firsts = await page.settle(t0, {"lines settled": lines_key, "panes complete": ct_key},
                                      {"first lines": lambda s: sum(s["lines"].values()) > 0})
    await record("add surface", t0, {"lines settled": times["lines settled"]}, firsts)
    lines_before = json.loads(lines_key(await page.state()))

    href = await page.address()
    t0 = time.monotonic()
    await page.open(href)
    times, firsts = await page.settle(t0, {"panes complete": ct_key, "lines settled": lines_key},
                                      {"first lines": lambda s: sum(s["lines"].values()) > 0})
    await record("reload", t0, times, firsts)
    lines_after = json.loads(lines_key(await page.state()))
    if lines_after != lines_before:
        failures.append(f"reload brought back other lines: {lines_after} after, {lines_before} before")

    xy = await page.pane_center("xy")
    await page.mouse("mouseMoved", *xy)
    await asyncio.sleep(0.2)
    position_before = await page.evaluate("document.querySelector('#position-number').value")
    await page.evaluate("window.__workflowFrames = 0; window.__workflowCounting = true; "
                        "(function count() { if (!window.__workflowCounting) return; window.__workflowFrames++; requestAnimationFrame(count) })(), true")
    task_before = {m["name"]: m["value"] for m in (await page.browser.send("Performance.getMetrics"))["metrics"]}["TaskDuration"]
    frames_before = (await page.state())["frames"]
    t0 = time.monotonic()
    await page.key("f", down_only=True)
    samples, next_sample, next_repeat = [], t0 + SAMPLE_S, t0 + KEY_REPEAT_S
    while time.monotonic() - t0 < HOLD_S:
        now = time.monotonic()
        if now >= next_repeat:
            await page.key("f", repeat=True, down_only=True)
            next_repeat += KEY_REPEAT_S
        if now >= next_sample:
            s = await page.state()
            reads, _ = chunk_reads(page.browser, t0, now)
            samples.append({"t": round(now - t0, 2), "frames": s["frames"] - frames_before, "chunk reads": reads, "share": display_share(s)})
            next_sample += SAMPLE_S
        await asyncio.sleep(0.005)
    await page.browser.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "f", "code": "KeyF", "windowsVirtualKeyCode": 70})
    t_release = time.monotonic()
    browser_frames = await page.evaluate("window.__workflowCounting = false, window.__workflowFrames")
    task_after = {m["name"]: m["value"] for m in (await page.browser.send("Performance.getMetrics"))["metrics"]}["TaskDuration"]
    frames = (await page.state())["frames"] - frames_before
    reads, mib = chunk_reads(page.browser, t0, t_release)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    times, _ = await page.settle(t_release, {"settle after release": ct_key})
    await record("held slice", t0, times)
    held_s = t_release - t0
    results["measures"].update({
        "held slice: viewer frames/s": round(frames / held_s, 1),
        "held slice: browser frames/s": round(browser_frames / held_s, 1),
        "held slice: main thread busy %": round(100 * (task_after - task_before) / held_s),
        "held slice: chunk reads": reads,
        "held slice: chunk MiB": mib,
    })
    for pane in samples[0]["share"]:
        shares = [sample["share"][pane] for sample in samples if sample["share"][pane] is not None]
        results["measures"][f"held slice: {pane} display-level share mean"] = round(sum(shares) / len(shares), 3)
    results["steps"][-1].update({"samples": samples, "position before": position_before})

    await page.type_into("position-number", CENTER_XYZ)
    t0 = time.monotonic()
    await page.key("Enter")
    times, _ = await page.settle(t0, {"panes complete": ct_key})
    await record("return to center", t0, times)

    for step, steps, delta in [("zoom in", WHEEL_IN_STEPS, -100), ("zoom out", WHEEL_OUT_STEPS, 100)]:
        await page.mouse("mouseMoved", *xy)
        t0 = time.monotonic()
        for _ in range(steps):
            await page.mouse("mouseWheel", *xy, deltaX=0, deltaY=delta)
            await asyncio.sleep(WHEEL_GAP_S)
        await page.mouse("mouseMoved", *NEUTRAL_XY)
        times, _ = await page.settle(t0, {"panes complete": ct_key})
        await record(step, t0, times)

    start = (xy[0] - DRAG_PX / 2, xy[1])
    await page.mouse("mouseMoved", *start)
    t0 = time.monotonic()
    await page.mouse("mousePressed", *start, button="left", buttons=1, clickCount=1)
    for i in range(1, DRAG_MOVES + 1):
        await page.mouse("mouseMoved", start[0] + DRAG_PX * i / DRAG_MOVES, start[1], button="left", buttons=1)
        await asyncio.sleep(FRAME_S)
    await page.mouse("mouseReleased", start[0] + DRAG_PX, start[1], button="left", buttons=0, clickCount=1)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    times, _ = await page.settle(t0, {"panes complete": ct_key})
    await record("pan", t0, times)

    await page.mouse("mouseMoved", *xy)
    await asyncio.sleep(0.2)
    t0 = time.monotonic()
    for _ in range(ROTATE_PRESSES):
        await page.key("ArrowRight")
        await asyncio.sleep(WHEEL_GAP_S)
    await page.mouse("mouseMoved", *NEUTRAL_XY)
    times, _ = await page.settle(t0, {"panes complete": ct_key})
    await record("rotation", t0, times)

    await page.evaluate("document.activeElement.blur(), true")
    t0 = time.monotonic()
    await page.key("u")
    times, _ = await page.settle(t0, {"complete": surface_key})
    await record("surface pane", t0, times)

    t0 = time.monotonic()
    await page.key("u")
    times, _ = await page.settle(t0, {"panes complete": ct_key})
    panes = await record("3d pane", t0, times)
    if panes["3d"]["centralColors"] < BLANK_COLORS:
        failures.append(f"the 3D pane is blank: {panes['3d']['centralColors']} colors in its central half, {out / panes['3d']['png']}")
    results["measures"]["3d pane: central colors"] = panes["3d"]["centralColors"]
    (out / "workflows.json").write_text(json.dumps(results, indent=1))
    print("\n".join(f"{name:52s} {value}" for name, value in results["measures"].items()))
    return failures
