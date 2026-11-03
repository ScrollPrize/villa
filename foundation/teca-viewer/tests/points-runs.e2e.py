import asyncio
import time

from page import CENTER_XYZ, POLL_S, VOLUME, ct_key

RUNS = 150
LAID_AT_MOST = 25
X0, Y0, Z0 = (round(float(value)) for value in CENTER_XYZ.split(","))
STEP_VOXELS = 4
REACHED_S = 5.0
NML = "runs.nml"
LISTED = """(() => {   // the list's count, its rows' run, label and whether marked, and whether each marked row shows
  const list = document.querySelector("#points-runs");
  const box = list.getBoundingClientRect();
  const rows = [...list.querySelectorAll("button")];
  return {
    count: document.querySelector("#points-runs-count").textContent,
    rows: rows.map((row) => [Number(row.dataset.run), row.firstElementChild.textContent, row.classList.contains("active")]),
    shown: rows.filter((row) => row.classList.contains("active")).map((row) => {
      const r = row.getBoundingClientRect();
      return r.top >= box.top && r.bottom <= box.bottom;
    }),
  };
})()"""


def nml():
    things = []
    for run in range(RUNS):
        x = X0 + run * STEP_VOXELS
        ids = [2 * run + 1, 2 * run + 2]
        nodes = "".join(f'<node id="{id}" radius="1" x="{x + i}" y="{Y0}" z="{Z0}"/>' for i, id in enumerate(ids))
        things.append(f'<thing id="{run + 1}" name="run {run + 1}"><nodes>{nodes}</nodes>'
                      f'<edges><edge source="{ids[0]}" target="{ids[1]}"/></edges></thing>')
    comments = "".join(f'<comment node="{id}" content="[KOLLESIS]"/>' for id in range(1, 2 * RUNS + 1))
    return (f'<?xml version="1.0" encoding="UTF-8"?><things><parameters><experiment name="runs"/></parameters>'
            f'{"".join(things)}<comments>{comments}</comments></things>')


async def check(page, out):
    failures = []
    t0 = time.monotonic()
    await page.open(f"?volume={VOLUME}")
    await page.settle(t0, {"panes complete": ct_key})
    await page.reveal("#points-section")
    (out / NML).write_text(nml())
    page.browser.downloads.mkdir(parents=True, exist_ok=True)
    (page.browser.downloads / NML).write_text(nml())
    await page.choose_file("#points-file", page.browser.downloads / NML)
    await page.wait_text("document.querySelector('#points-runs-count').textContent", lambda count: count == f"{RUNS} runs", 60)
    await page.evaluate("document.activeElement?.blur(), true")
    await page.key("6")

    async def reached(name, run):
        middle = [X0 + run * STEP_VOXELS + 1, Y0, Z0]
        t0 = time.monotonic()
        while True:
            got = await page.evaluate(LISTED)
            focus = [float(value) for value in (await page.evaluate("document.querySelector('#position-number').value")).split(",")]
            marked = [row[0] for row in got["rows"] if row[2]]
            if (got["count"] == f"{RUNS} runs" and len(got["rows"]) <= LAID_AT_MOST and marked == [run]
                    and got["shown"] == [True] and max(abs(a - b) for a, b in zip(focus, middle)) <= 1):
                return
            if time.monotonic() - t0 > REACHED_S:
                failures.append(f"after {name} the focus is {focus} and the list counts {got['count']!r}, lays out "
                                f"{len(got['rows'])} rows and marks {marked}, shown {got['shown']}, not run {run}'s middle "
                                f"{middle}, {RUNS} runs, at most {LAID_AT_MOST} and run {run} shown")
                return
            await asyncio.sleep(POLL_S)

    await page.evaluate("(() => { const list = document.querySelector('#points-runs'); list.scrollTop = list.scrollHeight; "
                        "return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))) })()")
    last = [row for row in (await page.evaluate(LISTED))["rows"] if row[0] == RUNS - 1]
    if last != [[RUNS - 1, f"run {RUNS} › kollesis", False]]:
        failures.append(f"scrolled to its end the list lays out {last} of its last run, not run {RUNS}'s row")
        return failures
    await page.click(f'#points-runs button[data-run="{RUNS - 1}"]')
    await reached("a click on the last run", RUNS - 1)
    await page.key(",")
    await reached(", from the last run", RUNS - 2)
    await page.evaluate("(() => { document.querySelector('#points-runs').scrollTop = 0; "
                        "return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))) })()")
    await page.key(".")
    await reached(". with the list scrolled to its start", RUNS - 1)
    return failures
