import unittest

from lasagna.scripts.bootstrap_venv import (
    environment_check_script,
    parse_cuda_version,
    select_backend,
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

    def test_check_script_runs_gpu_kernel_only_when_checking(self):
        self.assertIn("device='cuda'", environment_check_script("cu128", False))
        self.assertNotIn("device='cuda'", environment_check_script("cu128", True))
        self.assertNotIn("device='cuda'", environment_check_script("cpu", False))
        for backend, skip in (("cu128", False), ("cu128", True), ("cpu", False)):
            compile(environment_check_script(backend, skip), "<check>", "exec")


if __name__ == "__main__":
    unittest.main()
