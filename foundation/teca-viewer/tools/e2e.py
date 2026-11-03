# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp"]
# ///
import argparse
import asyncio
import importlib.util
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECK_TIMEOUT_S = 420   # tests/workflows.e2e.py, the slowest, takes about 90 s; a check past this has hung its page
CHECKS_AT_ONCE = 3      # the suite took 338 s, against 394 s at 4, where mesh-edit starves for download, and 771 s at 1
TIMED = {"workflows"}
PAGE = re.compile(r"index\.html|styles\.css|(src|vendor)/.+\.js|serve\.py|tools/(browser|page|e2e)\.py")
sys.path.insert(0, str(REPO / "tools"))
from browser import Browser  # noqa: E402
from page import HEIGHT, WIDTH, Page, serve  # noqa: E402


def staged(every):
    listed = subprocess.run(["git", "diff", "--cached", "--name-only", "--relative", "--no-renames", "-z"], cwd=REPO,
                            check=True, capture_output=True, text=True).stdout
    paths = listed.split("\0")[:-1]
    extensions = [path.stem for path in (REPO / "src" / "extensions").glob("*.js")]

    def owns(extension, check):
        return check == extension or check.startswith(f"{extension}-")

    def why(found):
        return f"{found[0]} staged" if len(found) == 1 else f"{found[0]} and {len(found) - 1} more staged"

    core = [path for path in paths if PAGE.fullmatch(path) and not path.startswith("src/extensions/")]
    if core:
        return every, f"all {len(every)} checks ({why(core)})"
    reached = {}
    for path in paths:
        parts = path.split("/")
        if parts[0] == "tests" and path.endswith(".e2e.py") and (REPO / path).is_file():
            check = parts[1].removesuffix(".e2e.py")
            name = next((extension for extension in extensions if owns(extension, check)), check)
            checks = {REPO / path}
        elif parts[:2] == ["src", "extensions"]:
            name = parts[2].removesuffix(".js")
            checks = {check for check in every if owns(name, check.name.removesuffix(".e2e.py"))}
        else:
            continue
        chosen, found = reached.setdefault(name, (set(), []))
        chosen.update(checks)
        found.append(path)
    if not reached:
        return [], "no browser check: no staged file is a check, an extension's, the page's or the runner's"
    return sorted(set().union(*(chosen for chosen, _ in reached.values()))), "; ".join(
        f"{name}: {len(chosen)} check{'s' if len(chosen) > 1 else ''} ({why(found)})"
        for name, (chosen, found) in reached.items())


async def main():
    parser = argparse.ArgumentParser(description="Browser checks on the real Paris 4 scroll.")
    parser.add_argument("checks", nargs="*", type=Path, help="check files; none runs every tests/*.e2e.py")
    parser.add_argument("--staged", action="store_true", help="the checks the files staged for commit reach")
    parser.add_argument("--web", type=Path, default=REPO, help="viewer tree whose serve.py serves it")
    parser.add_argument("--out", type=Path, default=Path("/tmp/teca-e2e"), help="directory for each check's out/<name>/")
    args = parser.parse_args()
    every = sorted((REPO / "tests").glob("*.e2e.py"))
    if args.staged and args.checks:
        parser.error("--staged chooses the checks; name none")
    if args.staged:
        checks, why = staged(every)
        print(why, flush=True)
        if not checks:
            return
    else:
        checks = [path.resolve() for path in args.checks] or every
    for path in checks:
        if not path.is_file():
            parser.error(f"no check file {path}")
    failed = []
    slots = asyncio.Semaphore(CHECKS_AT_ONCE)

    async def run(path, root):
        name = path.name.removesuffix(".e2e.py")
        spec = importlib.util.spec_from_file_location(f"check_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = args.out / name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        async with slots:
            print(f"{name}: running", flush=True)
            start = time.monotonic()
            async with Browser(WIDTH, HEIGHT) as browser:
                page = Page(browser, root)

                async def checked():
                    return await module.check(page, out) + await page.errors()

                try:
                    failures = await asyncio.wait_for(checked(), CHECK_TIMEOUT_S)
                except Exception as error:
                    failures = [f"{type(error).__name__}: {error}"] + [c["text"] for c in browser.console if c["level"] == "error"]
        print(f"{name}: {'failed' if failures else 'ok'} in {time.monotonic() - start:.0f} s", flush=True)
        for failure in failures:
            print(f"  {failure}", flush=True)
        if failures:
            kept = args.out / f"{name}-failed-{time.strftime('%Y%m%d-%H%M%S')}"
            await asyncio.to_thread(shutil.copytree, out, kept)
            print(f"  kept {kept}/", flush=True)
            failed.append(name)

    untimed = [path for path in checks if path.name.removesuffix(".e2e.py") not in TIMED]
    async with serve(args.web.resolve()) as root:
        await asyncio.gather(*(run(path, root) for path in untimed))
        for path in checks:
            if path not in untimed:
                await run(path, root)
    print(f"{len(checks) - len(failed)} of {len(checks)} checks ok" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


asyncio.run(main())
