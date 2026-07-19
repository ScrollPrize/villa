#!/usr/bin/env python3
"""Create a Lasagna venv with a driver-compatible PyTorch build."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TorchBuild:
    backend: str
    torch: str
    torchvision: str
    index_url: str


TORCH_BUILDS = {
    "cpu": TorchBuild(
        backend="cpu",
        torch="2.11.0",
        torchvision="0.26.0",
        index_url="https://download.pytorch.org/whl/cpu",
    ),
    "cu128": TorchBuild(
        backend="cu128",
        torch="2.11.0",
        torchvision="0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    ),
    "cu130": TorchBuild(
        backend="cu130",
        torch="2.13.0",
        torchvision="0.28.0",
        index_url="https://download.pytorch.org/whl/cu130",
    ),
}
CUDA13_ONLY_PACKAGES = {"cucim-cu13", "cupy-cuda13x", "nvidia-nvimgcodec-cu13"}


def parse_cuda_version(output: str) -> tuple[int, int] | None:
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def select_backend(cuda_version: tuple[int, int] | None) -> str:
    if cuda_version is None:
        return "cpu"
    if cuda_version >= (13, 0):
        return "cu130"
    if cuda_version >= (12, 8):
        return "cu128"
    raise RuntimeError(
        f"NVIDIA driver supports CUDA {cuda_version[0]}.{cuda_version[1]}; "
        "Lasagna requires a driver supporting CUDA 12.8 or newer"
    )


def detect_backend() -> tuple[str, str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "cpu", "nvidia-smi is not installed"
    result = subprocess.run(
        [nvidia_smi], capture_output=True, text=True, check=False
    )
    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi exists but cannot communicate with the driver; "
            "fix the driver or pass --backend cpu/cu128/cu130 explicitly"
        )
    cuda_version = parse_cuda_version(output)
    if cuda_version is None:
        raise RuntimeError("could not read 'CUDA Version' from nvidia-smi output")
    backend = select_backend(cuda_version)
    return backend, f"nvidia-smi reports CUDA {cuda_version[0]}.{cuda_version[1]}"


def run(command: list[str], *, dry_run: bool) -> None:
    import shlex

    print("$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def installed_packages(python: Path) -> set[str]:
    code = (
        "import importlib.metadata as m, json; "
        "print(json.dumps(sorted({d.metadata['Name'].lower() for d in m.distributions()})))"
    )
    result = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, check=True
    )
    return set(json.loads(result.stdout))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create/update a Lasagna Python 3.14 venv with compatible PyTorch."
    )
    parser.add_argument(
        "--venv", type=Path, default=Path(".venv"), help="Venv path (default: .venv)"
    )
    parser.add_argument("--python", default="3.14", help="Python version for uv venv")
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cu128", "cu130"),
        default="auto",
        help="PyTorch backend; auto inspects nvidia-smi",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Lasagna project directory",
    )
    parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Skip torch.cuda.is_available verification (for restricted sandboxes)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required; install it from https://docs.astral.sh/uv/")

    try:
        if args.backend == "auto":
            backend, reason = detect_backend()
        else:
            backend, reason = args.backend, "selected explicitly"
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    build = TORCH_BUILDS[backend]
    venv = args.venv.expanduser().resolve()
    project = args.project.resolve()
    venv_python = venv / "bin" / "python"
    print(f"PyTorch backend: {backend} ({reason})", flush=True)

    if not venv_python.exists():
        run([uv, "venv", "--python", args.python, str(venv)], dry_run=args.dry_run)

    if not args.dry_run and backend != "cu130":
        incompatible = sorted(CUDA13_ONLY_PACKAGES & installed_packages(venv_python))
        if incompatible:
            run(
                [uv, "pip", "uninstall", "--python", str(venv_python), *incompatible],
                dry_run=False,
            )

    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--index-url",
            build.index_url,
            f"torch=={build.torch}",
            f"torchvision=={build.torchvision}",
        ],
        dry_run=args.dry_run,
    )
    run(
        [uv, "pip", "install", "--python", str(venv_python), "-e", str(project)],
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0
    check = (
        "import json, torch; "
        "print(json.dumps({'torch': torch.__version__, "
        "'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()})); "
        f"raise SystemExit(0 if {backend == 'cpu' or args.skip_cuda_check!r} "
        "or torch.cuda.is_available() else 1)"
    )
    result = subprocess.run([str(venv_python), "-c", check], check=False)
    if result.returncode != 0:
        raise SystemExit(
            "PyTorch installed, but CUDA is unavailable. Check nvidia-smi/driver access "
            "or rerun with --backend cpu."
        )
    print(f"Lasagna environment ready: {venv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
