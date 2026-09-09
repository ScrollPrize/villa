import unittest
from pathlib import Path
from unittest.mock import patch

from lasagna.scripts.bootstrap_venv import (
    main, parse_cuda_version, select_backend,
    volume_cartographer_build_environment,
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
        ), patch("lasagna.scripts.bootstrap_venv.run") as mocked_run:
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

    @patch("lasagna.scripts.bootstrap_venv.sys.platform", "linux")
    @patch.dict("lasagna.scripts.bootstrap_venv.os.environ", {}, clear=True)
    def test_volume_cartographer_build_prefers_available_modern_gcc(self):
        def which(command):
            return f"/usr/bin/{command}" if command in {"gcc-13", "g++-13"} else None

        with patch("lasagna.scripts.bootstrap_venv.shutil.which", side_effect=which):
            environment = volume_cartographer_build_environment()

        self.assertEqual(environment["CC"], "/usr/bin/gcc-13")
        self.assertEqual(environment["CXX"], "/usr/bin/g++-13")


if __name__ == "__main__":
    unittest.main()
