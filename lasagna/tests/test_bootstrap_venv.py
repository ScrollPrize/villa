import unittest

from lasagna.scripts.bootstrap_venv import parse_cuda_version, select_backend


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


if __name__ == "__main__":
    unittest.main()
