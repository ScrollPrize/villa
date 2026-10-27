"""Check this machine for what the vesuvius tools need, and say how to fix what is missing.

    vesuvius.doctor             # every check, including the data servers
    vesuvius.doctor --offline   # skip the network checks
    vesuvius.doctor --json      # the same results for scripts

Exit status 1 when a check fails; warnings alone exit 0.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Sequence

DATA_SERVERS = {
    "dl.ash2txt.org": "https://dl.ash2txt.org/",
    "open-data bucket": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/?list-type=2&max-keys=1",
}

# Packages each group of commands imports, with the extra that installs them.
FEATURES = [
    ("volume access (Volume, zarr_tasks)", ["zarr", "numcodecs", "fsspec", "s3fs"], None),
    ("models (predict, train, compute_st)", ["torch", "scipy"], "models"),
    ("napari_trainer", ["napari", "magicgui", "qtpy"], "gui"),
]


@dataclass
class Check:
    name: str
    status: str  # ok, warn, fail
    detail: str
    fix: str = ""


def _version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_python() -> Check:
    have = platform.python_version()
    try:
        spec = md.metadata("vesuvius").get("Requires-Python") or ""
    except md.PackageNotFoundError:
        spec = ""
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        return Check("python", "ok", f"{have} (requirement {spec or 'unknown'} not checked)")
    if spec and have not in SpecifierSet(spec):
        return Check("python", "fail", f"{have}, but this vesuvius needs {spec}",
                     f"create an environment with a Python that matches {spec}, e.g. uv venv -p 3.14")
    return Check("python", "ok", f"{have}" + (f" (needs {spec})" if spec else ""))


def check_package() -> Check:
    ver = _version("vesuvius")
    if ver is None:
        return Check("vesuvius", "warn", "imported from a source tree, not installed",
                     "pip install -e vesuvius from the villa checkout, so the commands exist")
    return Check("vesuvius", "ok", ver)


def check_features() -> List[Check]:
    out = []
    for label, modules, extra in FEATURES:
        missing = []
        for mod in modules:
            try:
                importlib.import_module(mod)
            except Exception:  # an import that crashes counts as missing too
                missing.append(mod)
        if missing:
            fix = f'pip install "vesuvius[{extra}]"' if extra else "pip install " + " ".join(missing)
            out.append(Check(label, "warn" if extra else "fail", "missing " + ", ".join(missing), fix))
        else:
            vers = ", ".join(f"{m} {_version(m) or '?'}" for m in modules[:2])
            out.append(Check(label, "ok", vers))
    return out


def check_accelerator() -> Check:
    try:
        import torch
    except Exception:
        return Check("GPU", "warn", "torch not installed, so models run nowhere",
                     'pip install "vesuvius[models]"')
    if torch.cuda.is_available():
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        return Check("GPU", "ok", f"CUDA: {names}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return Check("GPU", "ok", "Apple MPS")
    return Check("GPU", "warn", "none found; models will run on the CPU, slowly",
                 "on NVIDIA, install a CUDA build of torch (see pytorch.org)")


def check_display() -> Check:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return Check("display", "ok", "none, QT_QPA_PLATFORM=offscreen")
        return Check("display", "warn", "no DISPLAY: napari_trainer and VC3D cannot open a window",
                     "use ssh -X, a desktop session, or xvfb-run for headless runs")
    return Check("display", "ok", os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform)


def check_cpus() -> Check:
    from vesuvius.image_proc.run.zarr_tasks.base import available_cpus

    usable, host = available_cpus(), os.cpu_count() or 1
    detail = f"{usable} usable of {host}"
    if usable < host:
        return Check("CPUs", "ok", detail + " (affinity or cgroup limit); worker defaults follow it")
    return Check("CPUs", "ok", detail)


def check_disk(path: str = ".") -> Check:
    free = shutil.disk_usage(path).free / 1e9
    if free < 5:
        return Check("disk", "warn", f"{free:.1f} GB free in {os.path.abspath(path)}",
                     "a scroll crop of 1024^3 voxels is 1 GB; free space or work elsewhere")
    return Check("disk", "ok", f"{free:.0f} GB free in {os.path.abspath(path)}")


def check_server(name: str, url: str, get: Optional[Callable] = None) -> Check:
    if get is None:
        import requests

        get = requests.get
    started = time.perf_counter()
    try:
        resp = get(url, timeout=8, stream=True)
        code = resp.status_code
        resp.close()
    except Exception as exc:  # report every network failure, with its type
        return Check(name, "fail", f"not reachable: {type(exc).__name__}: {exc}"[:200],
                     "check the network, proxy (HTTPS_PROXY) or firewall")
    ms = (time.perf_counter() - started) * 1000
    if code >= 400:
        return Check(name, "fail", f"HTTP {code} from {url}", "the server refused the request; try later")
    return Check(name, "ok" if ms < 1500 else "warn", f"HTTP {code} in {ms:.0f} ms",
                 "" if ms < 1500 else "slow link: use Volume(..., cache=True) or vesuvius.crop to work locally")


def run(offline: bool = False) -> List[Check]:
    checks = [check_python(), check_package(), *check_features(), check_accelerator(),
              check_display(), check_cpus(), check_disk()]
    if not offline:
        checks += [check_server(n, u) for n, u in DATA_SERVERS.items()]
    return checks


MARK = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="vesuvius.doctor", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("--offline", action="store_true", help="skip the data server checks")
    p.add_argument("--json", action="store_true", help="print the results as JSON")
    args = p.parse_args(argv)
    checks = run(offline=args.offline)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        width = max(len(c.name) for c in checks)
        for c in checks:
            print(f"{MARK[c.status]}  {c.name:<{width}}  {c.detail}")
            if c.fix and c.status != "ok":
                print(f"      fix: {c.fix}")
        fails = sum(c.status == "fail" for c in checks)
        warns = sum(c.status == "warn" for c in checks)
        print(f"\n{fails} failed, {warns} warning(s), {len(checks) - fails - warns} ok")
    return 1 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
