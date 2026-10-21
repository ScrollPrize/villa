import unittest
from pathlib import Path
from unittest.mock import patch

from lasagna.scripts.bootstrap_venv import (
    main, parse_cuda_version, select_backend,
    volume_cartographer_build_environment,
    environment_check_script,
)


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

    def test_dry_run_installs_sibling_volume_cartographer(self):
        villa = Path(__file__).resolve().parents[2]
        with patch(
            "lasagna.scripts.bootstrap_venv.shutil.which",
            return_value="/usr/bin/uv",
        ), patch("lasagna.scripts.bootstrap_venv.run") as mocked_run, patch(
            "lasagna.scripts.bootstrap_venv.volume_cartographer_build_environment",
            return_value={"CC": "/usr/bin/gcc-13", "CXX": "/usr/bin/g++-13"},
        ) as build_environment:
            self.assertEqual(main([
                "--project", str(villa / "lasagna"),
                "--venv", "/tmp/test-lasagna-bootstrap-venv",
                "--backend", "cpu",
                "--dry-run",
            ]), 0)

        commands = [call.args[0] for call in mocked_run.call_args_list]
        self.assertTrue(any(
            command[-2:] == ["-e", str(villa / "volume-cartographer")]
            for command in commands
        ))
        native_calls = [
            call for call in mocked_run.call_args_list
            if call.args[0][-2:] == ["-e", str(villa / "volume-cartographer")]
        ]
        self.assertEqual(len(native_calls), 1)
        self.assertEqual(native_calls[0].kwargs["environment"], build_environment.return_value)
        for call in mocked_run.call_args_list:
            if call not in native_calls:
                self.assertNotIn("environment", call.kwargs)

    @patch("lasagna.scripts.bootstrap_venv.sys.platform", "linux")
    @patch.dict("lasagna.scripts.bootstrap_venv.os.environ", {}, clear=True)
    def test_volume_cartographer_build_prefers_available_modern_gcc(self):
        def which(command):
            return f"/usr/bin/{command}" if command in {"gcc-13", "g++-13"} else None

        with patch("lasagna.scripts.bootstrap_venv.shutil.which", side_effect=which):
            environment = volume_cartographer_build_environment()

        self.assertEqual(environment["CC"], "/usr/bin/gcc-13")
        self.assertEqual(environment["CXX"], "/usr/bin/g++-13")

    def test_check_script_runs_gpu_kernel_only_when_checking(self):
        self.assertIn("device='cuda'", environment_check_script("cu128", False))
        self.assertNotIn("device='cuda'", environment_check_script("cu128", True))
        self.assertNotIn("device='cuda'", environment_check_script("cpu", False))
        for backend, skip in (("cu128", False), ("cu128", True), ("cpu", False)):
            compile(environment_check_script(backend, skip), "<check>", "exec")


if __name__ == "__main__":
    unittest.main()
