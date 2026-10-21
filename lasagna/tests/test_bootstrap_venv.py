import contextlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from lasagna.scripts.bootstrap_venv import (
    main, parse_cuda_version, select_backend,
    ROCM_HINT, TORCH_BUILDS, check_failure_message, detect_backend,
    detect_rocm, environment_check_script, rocm_hint,
)

MODULE = "lasagna.scripts.bootstrap_venv"
VILLA = Path(__file__).resolve().parents[2]
REAL_EXISTS = os.path.exists
REAL_IS_DIR = Path.is_dir


def which_only(*found):
    """shutil.which stand-in that finds only the named tools."""
    return lambda name: f"/usr/bin/{name}" if name in found else None


@contextlib.contextmanager
def rocm_files(*, opt_rocm=False, kfd=False, host=("linux", "x86_64")):
    """Pretend /opt/rocm and /dev/kfd exist or not, whatever the host has.

    host=(sys.platform, platform.machine()) to pretend; None keeps the caller's.
    """
    def is_dir(path):
        if path.as_posix() == "/opt/rocm":
            return opt_rocm
        return REAL_IS_DIR(path)

    def exists(path):
        if path == "/dev/kfd":
            return kfd
        return REAL_EXISTS(path)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(Path, "is_dir", is_dir))
        stack.enter_context(patch(f"{MODULE}.os.path.exists", side_effect=exists))
        if host is not None:
            stack.enter_context(patch(f"{MODULE}.sys.platform", host[0]))
            stack.enter_context(
                patch(f"{MODULE}.platform.machine", return_value=host[1])
            )
        yield


def dry_run(backend, *extra):
    """Run main() as a dry run; return (exit code, commands, stdout)."""
    stdout = io.StringIO()
    with patch(f"{MODULE}.run") as mocked_run, contextlib.redirect_stdout(stdout):
        code = main([
            "--project", str(VILLA / "lasagna"),
            "--venv", "/tmp/test-lasagna-bootstrap-venv",
            "--backend", backend, "--dry-run", *extra,
        ])
    return code, [call.args[0] for call in mocked_run.call_args_list], stdout.getvalue()


def torch_install(commands):
    return next(command for command in commands if "--index-url" in command)


class BootstrapVenvTests(unittest.TestCase):
    def test_parse_cuda_version(self):
        output = "NVIDIA-SMI 570.00  Driver Version: 570.00  CUDA Version: 12.8"
        self.assertEqual(parse_cuda_version(output), (12, 8))

    def test_backend_selection(self):
        self.assertEqual(select_backend(None), "cpu")
        self.assertEqual(select_backend((12, 8)), "cu128")
        self.assertEqual(select_backend((12, 9)), "cu128")
        self.assertEqual(select_backend((13, 0)), "cu130")

    def test_old_driver_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA 12.8 or newer"):
            select_backend((12, 7))

    def test_auto_never_selects_rocm_when_rocm_tools_are_found(self):
        with patch(f"{MODULE}.shutil.which", side_effect=which_only("rocminfo")), \
                rocm_files():
            self.assertEqual(detect_backend(), ("cpu", "nvidia-smi is not installed"))
            self.assertEqual(detect_rocm(), "rocminfo found")
            self.assertEqual(rocm_hint(), ROCM_HINT)

    def test_opt_rocm_alone_is_a_rocm_sign(self):
        with patch(f"{MODULE}.shutil.which", return_value=None), \
                rocm_files(opt_rocm=True):
            self.assertEqual(detect_rocm(), "/opt/rocm exists")
            self.assertEqual(detect_backend()[0], "cpu")
            self.assertEqual(rocm_hint(), ROCM_HINT)

    def test_dev_kfd_alone_is_a_rocm_sign(self):
        with patch(f"{MODULE}.shutil.which", return_value=None), rocm_files(kfd=True):
            self.assertEqual(detect_rocm(), "/dev/kfd exists")

    def test_nothing_found_gives_cpu_without_hint(self):
        with patch(f"{MODULE}.shutil.which", return_value=None), rocm_files():
            self.assertEqual(detect_backend(), ("cpu", "nvidia-smi is not installed"))
            self.assertIsNone(detect_rocm())
            self.assertIsNone(rocm_hint())

    def test_working_nvidia_smi_is_unchanged_by_rocm_tools(self):
        def nvidia_smi(command, **kwargs):
            if "--query-gpu=compute_cap" in command:
                stdout = "8.9\n"
            else:
                stdout = "CUDA Version: 13.0"
            return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        with patch(f"{MODULE}.shutil.which",
                   side_effect=which_only("nvidia-smi", "rocminfo")), \
                patch(f"{MODULE}.subprocess.run", side_effect=nvidia_smi):
            self.assertEqual(
                detect_backend(), ("cu130", "nvidia-smi reports CUDA 13.0")
            )

    def test_no_hint_where_rocm_cannot_install(self):
        for host in (("linux", "aarch64"), ("darwin", "arm64")):
            with self.subTest(host=host),                     patch(f"{MODULE}.shutil.which", side_effect=which_only("rocminfo")),                     rocm_files(opt_rocm=True, kfd=True, host=host):
                self.assertEqual(detect_backend()[0], "cpu")
                self.assertEqual(detect_rocm(), "rocminfo found")
                self.assertIsNone(rocm_hint())

    def test_auto_with_rocm_tools_installs_cpu_and_prints_hint(self):
        with patch(f"{MODULE}.shutil.which", side_effect=which_only("uv", "rocminfo")), \
                rocm_files(opt_rocm=True, kfd=True):
            code, commands, stdout = dry_run("auto")
        self.assertEqual(code, 0)
        self.assertIn("PyTorch backend: cpu (nvidia-smi is not installed)", stdout)
        self.assertIn(ROCM_HINT, stdout)
        self.assertIn(TORCH_BUILDS["cpu"].index_url, torch_install(commands))

    def test_auto_without_rocm_signs_prints_no_hint(self):
        with patch(f"{MODULE}.shutil.which", side_effect=which_only("uv")), rocm_files():
            code, commands, stdout = dry_run("auto")
        self.assertEqual(code, 0)
        self.assertNotIn(ROCM_HINT, stdout)
        self.assertIn(TORCH_BUILDS["cpu"].index_url, torch_install(commands))

    def test_broken_nvidia_smi_on_amd_box_keeps_old_error_plus_hint(self):
        broken = types.SimpleNamespace(
            returncode=9, stdout="", stderr="NVIDIA-SMI has failed"
        )
        with patch(f"{MODULE}.shutil.which",
                   side_effect=which_only("uv", "nvidia-smi", "rocminfo")), \
                patch(f"{MODULE}.subprocess.run", return_value=broken), \
                rocm_files(), self.assertRaises(SystemExit) as raised:
            dry_run("auto")
        message = str(raised.exception.code)
        self.assertIn("cannot communicate with the driver", message)
        self.assertIn(ROCM_HINT, message)

    @patch(f"{MODULE}.sys.platform", "linux")
    @patch(f"{MODULE}.platform.machine", return_value="x86_64")
    def test_explicit_rocm_on_linux_x86_64(self, _machine):
        with patch(f"{MODULE}.shutil.which", side_effect=which_only("uv")), \
                rocm_files(kfd=True):
            code, commands, stdout = dry_run("rocm")
        self.assertEqual(code, 0)
        self.assertIn("PyTorch backend: rocm (selected explicitly)", stdout)
        install = torch_install(commands)
        self.assertIn("https://download.pytorch.org/whl/rocm7.2", install)
        self.assertIn("torch==2.13.0", install)
        self.assertIn("torchvision==0.28.0", install)

    def test_explicit_rocm_refused_on_other_os_or_arch(self):
        for system, machine in (("linux", "aarch64"), ("darwin", "arm64"),
                                ("darwin", "x86_64"), ("win32", "AMD64")):
            with self.subTest(system=system, machine=machine), \
                    patch(f"{MODULE}.sys.platform", system), \
                    patch(f"{MODULE}.platform.machine", return_value=machine), \
                    patch(f"{MODULE}.shutil.which", side_effect=which_only("uv")), \
                    rocm_files(kfd=True, host=None),                     patch(f"{MODULE}.run") as mocked_run:
                with self.assertRaises(SystemExit) as raised:
                    main(["--backend", "rocm", "--dry-run"])
                self.assertIn("needs Linux x86_64", str(raised.exception.code))
                mocked_run.assert_not_called()

    @patch(f"{MODULE}.sys.platform", "linux")
    @patch(f"{MODULE}.platform.machine", return_value="x86_64")
    def test_explicit_rocm_needs_dev_kfd_unless_check_skipped(self, _machine):
        with patch(f"{MODULE}.shutil.which", side_effect=which_only("uv")), \
                rocm_files(opt_rocm=True):
            with self.assertRaises(SystemExit) as raised:
                dry_run("rocm")
            self.assertIn("/dev/kfd does not exist", str(raised.exception.code))
            code, commands, _ = dry_run("rocm", "--skip-cuda-check")
        self.assertEqual(code, 0)
        self.assertIn("torch==2.13.0", torch_install(commands))

    def test_rocm_pin_matches_cu130(self):
        self.assertEqual(TORCH_BUILDS["rocm"].torch, TORCH_BUILDS["cu130"].torch)
        self.assertEqual(TORCH_BUILDS["rocm"].torchvision,
                         TORCH_BUILDS["cu130"].torchvision)

    def test_failure_messages_are_backend_aware(self):
        self.assertIn("nvidia-smi", check_failure_message("cu128", 1))
        unavailable = check_failure_message("rocm", 1)
        for text in ("/dev/kfd", "'render'", "'video'", "HSA_OVERRIDE_GFX_VERSION"):
            self.assertIn(text, unavailable)
        self.assertNotIn("nvidia-smi", unavailable)
        self.assertIn("HSA_OVERRIDE_GFX_VERSION", check_failure_message("rocm", 2))

    def test_check_script_runs_gpu_kernel_only_when_checking(self):
        self.assertIn("device='cuda'", environment_check_script("cu128", False))
        self.assertNotIn("device='cuda'", environment_check_script("cu128", True))
        self.assertNotIn("device='cuda'", environment_check_script("cpu", False))
        self.assertIn("gcnArchName", environment_check_script("rocm", False))
        self.assertIn("ROCm smoke test failed", environment_check_script("rocm", False))
        self.assertNotIn("gcnArchName", environment_check_script("cu130", False))
        for backend, skip in (("cu128", False), ("cu128", True), ("cpu", False),
                              ("rocm", False), ("rocm", True)):
            compile(environment_check_script(backend, skip), "<check>", "exec")

    def run_rocm_check(self, gpu_arch, environ=None):
        """Execute the rocm check script on a fake torch; return (exit code, stdout)."""
        class Tensor:
            def __mul__(self, other):
                return self

            def sum(self):
                return self

            def item(self):
                return 16.0

        torch = types.ModuleType("torch")
        torch.__version__ = "2.13.0+rocm7.2"
        torch.version = types.SimpleNamespace(cuda=None, hip="7.2.0")
        torch.ones = lambda *args, **kwargs: Tensor()
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_arch_list=lambda: ["gfx1100", "gfx1201", "gfx90a:xnack-"],
            get_device_properties=lambda index: types.SimpleNamespace(
                gcnArchName=gpu_arch
            ),
        )
        modules = {"torch": torch}
        parent = None
        for name in ("vesuvius", "vesuvius.neural_tracing",
                     "vesuvius.neural_tracing.fiber_trace_3d",
                     "vesuvius.neural_tracing.fiber_trace_3d.infer"):
            module = types.ModuleType(name)
            module.__file__ = f"/fake/{name}.py"
            if parent is not None:
                setattr(parent, name.rsplit(".", 1)[1], module)
            modules[name] = parent = module
        stdout = io.StringIO()
        with patch.dict(sys.modules, modules), \
                patch.dict(os.environ, environ or {}, clear=True), \
                patch.object(Path, "is_file", return_value=True), \
                contextlib.redirect_stdout(stdout), \
                self.assertRaises(SystemExit) as raised:
            exec(environment_check_script("rocm", False), {})
        return raised.exception.code, stdout.getvalue()

    def test_rocm_check_accepts_gfx_target_in_wheel(self):
        self.assertEqual(self.run_rocm_check("gfx1201")[0], 0)
        self.assertEqual(self.run_rocm_check("gfx90a:sramecc+:xnack-")[0], 0)

    def test_rocm_check_rejects_gfx_target_missing_from_wheel(self):
        code, stdout = self.run_rocm_check("gfx1030")
        self.assertEqual(code, 2)
        self.assertIn("0:gfx1030", stdout)
        self.assertIn("gfx1201", stdout)

    def test_rocm_check_does_not_block_hsa_override(self):
        code, stdout = self.run_rocm_check(
            "gfx1030", environ={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}
        )
        self.assertEqual(code, 0)
        self.assertIn("HSA_OVERRIDE_GFX_VERSION is set", stdout)


if __name__ == "__main__":
    unittest.main()
