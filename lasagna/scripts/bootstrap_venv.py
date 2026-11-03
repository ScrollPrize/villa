#!/usr/bin/env python3
"""Create a Lasagna venv with a driver-compatible PyTorch build."""
from __future__ import annotations

import argparse
import json
import os
import platform
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


# Include the wheel backend in GPU pins so an installed build for another
# backend cannot satisfy the requirement when reusing a venv.
TORCH_BUILDS = {
    "cpu": TorchBuild(
        backend="cpu",
        torch="2.11.0",
        torchvision="0.26.0",
        index_url="https://download.pytorch.org/whl/cpu",
    ),
    "cu128": TorchBuild(
        backend="cu128",
        torch="2.11.0+cu128",
        torchvision="0.26.0+cu128",
        index_url="https://download.pytorch.org/whl/cu128",
    ),
    "cu130": TorchBuild(
        backend="cu130",
        torch="2.13.0+cu130",
        torchvision="0.28.0+cu130",
        index_url="https://download.pytorch.org/whl/cu130",
    ),
    # Opt-in only (--backend rocm); auto never selects it. The rocm7.2 index
    # ships manylinux_2_28_x86_64 wheels only.
    "rocm": TorchBuild(
        backend="rocm",
        torch="2.13.0+rocm7.2",
        torchvision="0.28.0+rocm7.2",
        index_url="https://download.pytorch.org/whl/rocm7.2",
    ),
}
CUDA13_ONLY_PACKAGES = {"cucim-cu13", "cupy-cuda13x", "nvidia-nvimgcodec-cu13"}

# Lowest CUDA compute capability the official wheels carry SASS for. The cu128
# and cu130 builds start at sm_75, so a Pascal or Volta card installs cleanly
# and then fails at the first kernel launch with
# "no kernel image is available for execution on the device".
BACKEND_MIN_COMPUTE_CAP = {"cu128": (7, 5), "cu130": (7, 5)}


def venv_interpreter(venv: Path) -> Path:
    """Interpreter inside `venv`: Scripts/python.exe on Windows, bin/python elsewhere."""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


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


def parse_compute_caps(output: str) -> list[tuple[int, int]]:
    caps = []
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", line)
        if match is not None:
            caps.append((int(match.group(1)), int(match.group(2))))
    return caps


def query_compute_caps(nvidia_smi: str) -> list[tuple[int, int]]:
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return parse_compute_caps(result.stdout)


ROCM_DEVICE = "/dev/kfd"
ROCM_HINT = "AMD ROCm tools found; to use an AMD GPU run with --backend rocm"


def detect_rocm() -> str | None:
    """Describe the ROCm signs found on this machine, or None.

    Tools or /opt/rocm do not prove that a usable AMD GPU is present (leftover
    packages, a container without the device passed through), so auto
    detection only uses this to print ROCM_HINT and never selects rocm.
    """
    for tool in ("rocminfo", "amd-smi", "rocm-smi"):
        if shutil.which(tool) is not None:
            return f"{tool} found"
    if Path("/opt/rocm").is_dir():
        return "/opt/rocm exists"
    if os.path.exists(ROCM_DEVICE):
        return f"{ROCM_DEVICE} exists"
    return None


def rocm_hint() -> str | None:
    """One-line hint for --backend auto when ROCm signs are present.

    Not shown where --backend rocm would be refused (not Linux x86_64).
    """
    if detect_rocm() is None or rocm_unsupported_reason(need_device=False):
        return None
    return ROCM_HINT


def rocm_unsupported_reason(*, need_device: bool) -> str | None:
    """Why --backend rocm cannot work here, or None if it can be tried."""
    machine = platform.machine()
    if sys.platform != "linux" or machine != "x86_64":
        return (
            f"--backend rocm needs Linux x86_64 (the rocm7.2 index only has "
            f"manylinux_2_28_x86_64 wheels); this machine is {sys.platform} "
            f"{machine or 'unknown'}. Use --backend cpu"
        )
    if need_device and not os.path.exists(ROCM_DEVICE):
        return (
            f"--backend rocm: {ROCM_DEVICE} does not exist, so no AMD GPU is "
            "visible (amdgpu kernel driver not loaded, or the device was not "
            "passed into this container: --device /dev/kfd --device /dev/dri). "
            "Use --backend cpu, or --skip-cuda-check to install without a GPU"
        )
    return None


def detect_backend() -> tuple[str, str]:
    """Pick cpu, cu128 or cu130 from nvidia-smi. rocm is opt-in only."""
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
            "fix the driver or pass --backend cpu/cu128/cu130/rocm explicitly"
        )
    cuda_version = parse_cuda_version(output)
    if cuda_version is None:
        raise RuntimeError("could not read 'CUDA Version' from nvidia-smi output")
    backend = select_backend(cuda_version)
    reason = f"nvidia-smi reports CUDA {cuda_version[0]}.{cuda_version[1]}"

    # The driver's CUDA version says nothing about whether the wheels contain
    # kernels for this GPU. Check the card itself before picking a CUDA build.
    minimum = BACKEND_MIN_COMPUTE_CAP.get(backend)
    caps = query_compute_caps(nvidia_smi) if minimum is not None else []
    if minimum is not None and caps and max(caps) < minimum:
        best = max(caps)
        return "cpu", (
            f"{reason}, but the newest GPU is compute capability "
            f"{best[0]}.{best[1]} and the {backend} wheels start at "
            f"{minimum[0]}.{minimum[1]}; falling back to cpu. Pass "
            f"--backend {backend} to override"
        )
    return backend, reason


def run(
    command: list[str], *, dry_run: bool, environment: dict[str, str] | None = None,
) -> None:
    import shlex

    assignments = []
    if environment is not None:
        assignments = [
            f"{name}={environment[name]}"
            for name in ("CC", "CXX")
            if environment.get(name) != os.environ.get(name)
        ]
    prefix = (shlex.join(assignments) + " ") if assignments else ""
    print("$ " + prefix + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=environment)


def volume_cartographer_build_environment() -> dict[str, str] | None:
    """Select a sufficiently modern GNU C++ compiler on Linux when available."""
    if not sys.platform.startswith("linux") or os.environ.get("CC") or os.environ.get("CXX"):
        return None
    for version in (15, 14, 13):
        cc = shutil.which(f"gcc-{version}")
        cxx = shutil.which(f"g++-{version}")
        if cc is not None and cxx is not None:
            return {**os.environ, "CC": cc, "CXX": cxx}
    return None


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
        choices=("auto", "cpu", "cu128", "cu130", "rocm"),
        default="auto",
        help="PyTorch backend; auto inspects nvidia-smi (cpu/cu128/cu130); "
        "rocm (AMD, Linux x86_64) is opt-in only",
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


def environment_check_script(backend: str, skip_cuda_check: bool) -> str:
    """Python source run inside the new venv to report the install.

    torch.cuda.is_available() only reports that a driver and a device are
    present; it stays True on a GPU whose compute capability the wheel has
    no kernels for. Launch one real kernel so a mismatch surfaces here
    rather than part-way through a fit. With --skip-cuda-check or the CPU
    backend the GPU is not touched at all.
    """
    lines = [
        "import json, torch",
        "from pathlib import Path",
        "import sys",
        "vc_fiberlets = Path(sys.executable).with_name('vc_fiberlets.exe' if sys.platform == 'win32' else 'vc_fiberlets')",
        "assert vc_fiberlets.is_file(), 'vc_fiberlets is missing from the venv'",
        "import vesuvius.neural_tracing.fiber_trace_3d.infer as fiber_infer",
        "print(json.dumps({'torch': torch.__version__,"
        " 'torch_cuda': torch.version.cuda, 'torch_hip': torch.version.hip,"
        " 'cuda_available': torch.cuda.is_available(),"
        " 'fiber_infer': fiber_infer.__file__}))",
    ]
    if backend == "rocm":
        lines.append("assert torch.version.hip, 'Expected a ROCm PyTorch build'")
    if backend != "cpu" and not skip_cuda_check:
        api = "ROCm" if backend == "rocm" else "CUDA"
        lines += [
            "ok = torch.cuda.is_available()",
            "if ok:",
            "    try:",
            "        (torch.ones(8, device='cuda') * 2).sum().item()",
            "    except Exception as exc:",
            f"        print('{api} smoke test failed:', exc)",
        ]
        if backend == "rocm":
            lines += [
                "        try:",
                "            print('GPU gfx target:', torch.cuda.get_device_properties(0).gcnArchName,",
                "                  '| wheel targets:', ' '.join(torch.cuda.get_arch_list()))",
                "        except Exception as diagnostic_exc:",
                "            print('Could not query ROCm targets:', diagnostic_exc)",
            ]
        lines += [
            "        ok = False",
            "raise SystemExit(0 if ok else 1)",
        ]
    return "\n".join(lines)


def check_failure_message(backend: str) -> str:
    """Error for a failed environment check, worded for the installed backend."""
    if backend != "rocm":
        return (
            "PyTorch installed, but CUDA is unavailable. Check nvidia-smi/driver access "
            "or rerun with --backend cpu."
        )
    return (
        "PyTorch installed, but the AMD GPU is unavailable through ROCm. Check that "
        "/dev/kfd and /dev/dri/renderD* are accessible (add your user to the "
        "'render' and 'video' groups and log in again; in a container pass "
        "--device /dev/kfd --device /dev/dri), and that HSA_OVERRIDE_GFX_VERSION "
        "is unset or names a target your GPU supports; or rerun with --backend cpu."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required; install it from https://docs.astral.sh/uv/")

    if args.backend == "rocm":
        unsupported = rocm_unsupported_reason(need_device=not args.skip_cuda_check)
        if unsupported is not None:
            raise SystemExit(unsupported)
    hint = rocm_hint() if args.backend == "auto" else None
    try:
        if args.backend == "auto":
            backend, reason = detect_backend()
        else:
            backend, reason = args.backend, "selected explicitly"
    except RuntimeError as exc:
        raise SystemExit(str(exc) + (f"\n{hint}" if hint else "")) from exc

    build = TORCH_BUILDS[backend]
    venv = args.venv.expanduser().resolve()
    project = args.project.resolve()
    venv_python = venv_interpreter(venv)
    volume_cartographer = project.parent / "volume-cartographer"
    if not (volume_cartographer / "pyproject.toml").is_file():
        raise SystemExit(
            "the Lasagna bootstrap requires the sibling volume-cartographer checkout"
        )
    object_store = project.parent / "vc_object_store"
    if not (object_store / "pyproject.toml").is_file():
        raise SystemExit(
            "the Lasagna bootstrap requires the sibling vc_object_store checkout"
        )
    print(f"PyTorch backend: {backend} ({reason})", flush=True)
    if hint is not None:
        print(hint, flush=True)

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
        [
            uv,
            "pip",
            "install",
            "--python",
            str(venv_python),
            "-e",
            str(volume_cartographer),
        ],
        dry_run=args.dry_run,
        environment=volume_cartographer_build_environment(),
    )
    run(
        [uv, "pip", "install", "--python", str(venv_python), "-e", str(object_store)],
        dry_run=args.dry_run,
        environment=volume_cartographer_build_environment(),
    )
    run(
        [uv, "pip", "install", "--python", str(venv_python), "-e", str(project)],
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0
    check = environment_check_script(backend, args.skip_cuda_check)
    check_environment = dict(os.environ)
    check_environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [str(venv_python), "-c", check], check=False, cwd=venv,
        env=check_environment,
    )
    if result.returncode != 0:
        raise SystemExit(check_failure_message(backend))
    print(f"Lasagna environment ready: {venv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
