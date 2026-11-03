# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp"]
# ///
import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser import Browser, find_free_port  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VOLUME = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
OPEN_TIMEOUT_S = 60

PAGE = """(() => {
  const text = node => node.textContent.replace(/\\s+/g, ' ').trim()
  const label = c => c.tagName === 'BUTTON' ? text(c) || c.getAttribute('aria-label') || ''
    : text(document.querySelector(`label[for="${c.id}"]`) || c.closest('label') || c)
  const controls = root => [...root.querySelectorAll('input, select, button, textarea, output')].filter(c => c.id !== '')
    .map(c => [c.id, c.tagName.toLowerCase() + (c.tagName === 'INPUT' ? ':' + c.type : ''), label(c).slice(0, 60), c.title || ''])
  const heading = section => section.querySelector('h2, h3, .card-title, summary')
  return {
    keys: [...document.querySelectorAll('#shortcuts dt')].map(dt => [text(dt), text(dt.nextElementSibling), text(dt.nextElementSibling.nextElementSibling)]),
    top: controls(document.querySelector('#top-bar')),
    panels: [...document.querySelectorAll('.side-panel section.card')].map(section => ({
      heading: section.id || (heading(section) ? text(heading(section)) : ''), hidden: section.hidden, controls: controls(section),
      tab: section.closest('[data-lpanel]')?.dataset.lpanel.toUpperCase() ??
        document.querySelector(`[data-rtab="${section.closest('[data-rpanel]').dataset.rpanel}"]`).title + ' tab',
    })),
    panes: [...document.querySelectorAll('#ortho-grid [id^="viewer-"]')].map(pane => [pane.id, pane.hidden]),
    dialogs: [...document.querySelectorAll('.modal-backdrop, dialog')].map(dialog => ({ id: dialog.id, controls: controls(dialog) })),
    status: [...document.querySelectorAll('#status-bar [id]')].map(node => node.id),
  }
})()"""


def rows(controls):
    for control_id, kind, name, tip in controls:
        print(f"  #{control_id:<34}{kind:<16}{name}  {tip}".rstrip())


async def main():
    parser = argparse.ArgumentParser(description="Print the keys, controls and panes of the running viewer.")
    parser.add_argument("--query", default=f"?volume={VOLUME}", help="the address query the page opens with")
    args = parser.parse_args()
    port = find_free_port()
    server = subprocess.Popen([sys.executable, str(REPO / "serve.py"), "--port", str(port)], stdout=subprocess.DEVNULL)
    try:
        async with Browser(width=1440, height=1000) as browser:
            await browser.navigate(f"http://127.0.0.1:{port}/{args.query}")
            await browser.wait_for("document.querySelector('#ortho-grid [id^=\"viewer-\"]') !== null", OPEN_TIMEOUT_S)
            page = await browser.evaluate(PAGE)
    finally:
        server.terminate()
        server.wait()

    print("KEYS (the shortcuts dialog)")
    for keys, description, held in page["keys"]:
        print(f"  {keys:<28}{description:<74}{held}".rstrip())
    print("\nTOP BAR")
    rows(page["top"])
    print("\nSIDE PANELS")
    tab = None
    for section in page["panels"]:
        if section["tab"] != tab:
            tab = section["tab"]
            print(f" {tab}")
        print(f"  {section['heading']}{'  (hidden until a Surface pane opens)' if section['hidden'] else ''}")
        rows(section["controls"])
    print("\nPANES\n  " + "  ".join(f"#{pane_id}{' (hidden)' if hidden else ''}" for pane_id, hidden in page["panes"]))
    print("\nDIALOGS")
    for dialog in page["dialogs"]:
        print(f" #{dialog['id']}")
        rows(dialog["controls"])
    print("\nSTATUS BAR\n  " + "  ".join(f"#{node_id}" for node_id in page["status"]))
    main_source = (REPO / "src/main.js").read_text()
    session_source = (REPO / "src/session.js").read_text()
    query = sorted(set(re.findall(r"query\.(?:get|getAll|has)\(\"(\w+)\"\)", main_source)))
    fields = re.findall(r"\"(\w+)\"", re.search(r"SESSION_KEYS = \[([^\]]*)\]", session_source)[1])
    print("\nADDRESS (src/main.js, src/session.js)\n  query: " + "  ".join(query) + "\n  #session: " + "  ".join(fields))


asyncio.run(main())
