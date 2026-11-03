import argparse
import asyncio
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECK_TIMEOUT_S = 300
BROWSERS = 6            # more gains little over the slowest check, twelve together can leave a page that never answers, and on the GPU one page's large upload can hang the driver for all
sys.path.insert(0, str(REPO / "tools"))
from browser import Browser, find_free_port  # noqa: E402


async def main():
    parser = argparse.ArgumentParser(description="Browser checks in headless Chromium")
    parser.add_argument("checks", nargs="*", type=Path, help="check files; none runs every tests/*.e2e.py")
    args = parser.parse_args()
    checks = [path.resolve() for path in args.checks] or sorted((REPO / "tests").glob("*.e2e.py"))
    if any(path.name in ("real.e2e.py", "extend.e2e.py") for path in checks) and not (REPO / "real/probes.json").exists():
        subprocess.run([sys.executable, str(REPO / "tools/real_data.py"), "--out", str(REPO / "real")], check=True)
    port = find_free_port()
    server = subprocess.Popen([sys.executable, str(REPO / "tools/serve.py"), "--port", str(port), "--dir", str(REPO)])
    modules = []
    for path in checks:
        spec = importlib.util.spec_from_file_location(f"check_{path.name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append((path.name.removesuffix(".e2e.py"), module))

    slots = asyncio.Semaphore(BROWSERS)

    async def run(name, module):
        stage = "starting its browser"

        async def checked():
            nonlocal stage
            async with Browser() as browser:
                started = time.monotonic()
                stage = "opening its page"
                await browser.navigate(f"http://127.0.0.1:{port}/{module.HASH}")
                await browser.wait_for("window.vesuvia", 30)
                stage = "checking"
                failures = [f"{name}: {failure}" for failure in await module.check(browser)]
                errors = [f"{name}: {line['text']}" for line in browser.console if line["level"] == "error"]
                print(f"{name} checked in {time.monotonic() - started:.1f} s", flush=True)
                return failures, errors

        async with slots:
            running = asyncio.ensure_future(checked())
            if not (await asyncio.wait({running}, timeout=CHECK_TIMEOUT_S))[0]:
                running.cancel()
                await asyncio.wait({running})
                print(f"{name} did not finish in {CHECK_TIMEOUT_S} s, {stage}", flush=True)
                return [f"{name}: did not finish in {CHECK_TIMEOUT_S} s, {stage}"], []
            return running.result()

    try:
        results = await asyncio.gather(*(run(name, module) for name, module in modules))
    finally:
        server.terminate()
        server.wait()
    failures = [failure for found, _ in results for failure in found]
    errors = [error for _, found in results for error in found]
    for text in errors:
        print("console error:", text)
    for text in failures:
        print("FAIL", text)
    print("ok" if not failures and not errors else f"{len(failures)} failures, {len(errors)} console errors")
    return 1 if failures or errors else 0


sys.exit(asyncio.run(main()))
