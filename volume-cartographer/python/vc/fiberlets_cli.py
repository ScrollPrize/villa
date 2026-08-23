"""Console-script bridge to the packaged native ``vc_fiberlets`` binary."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def packaged_binary() -> Path:
    name = "vc_fiberlets.exe" if os.name == "nt" else "vc_fiberlets"
    return Path(__file__).resolve().parent / "bin" / name


def main() -> int:
    binary = packaged_binary()
    if not binary.is_file():
        raise FileNotFoundError(
            f"the volume-cartographer installation is missing {binary}; "
            "reinstall it with Fiberlet CLI support"
        )
    if os.name == "nt":
        # The packaged DLLs live at the site-packages root, two directories
        # above the private executable. Propagate that search path to Windows.
        site_packages = str(binary.parent.parent.parent)
        os.environ["PATH"] = site_packages + os.pathsep + os.environ.get("PATH", "")
    os.execv(str(binary), [str(binary), *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
