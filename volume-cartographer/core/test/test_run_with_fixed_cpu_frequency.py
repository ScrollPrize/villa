import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_with_fixed_cpu_frequency.py"
)
SPEC = importlib.util.spec_from_file_location("run_with_fixed_cpu_frequency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FixedCpuFrequencyTest(unittest.TestCase):
    def make_sysfs(self, root: Path) -> Path:
        cpu = root / "cpu"
        cpufreq = cpu / "cpufreq"
        cpufreq.mkdir(parents=True)
        (cpufreq / "boost").write_text("1\n")
        for index in range(2):
            policy = cpufreq / f"policy{index}"
            policy.mkdir()
            (policy / "scaling_governor").write_text("powersave\n")
            (policy / "energy_performance_preference").write_text(
                "balance_performance\n"
            )
            (policy / "scaling_min_freq").write_text("1700000\n")
            (policy / "scaling_max_freq").write_text("5000000\n")
            (policy / "scaling_cur_freq").write_text("3000000\n")
            cppc = cpu / f"cpu{index}" / "acpi_cppc"
            cppc.mkdir(parents=True)
            (cppc / "nominal_freq").write_text("3401\n")
        return cpu

    def test_apply_verify_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sysfs = self.make_sysfs(root)
            recovery = root / "state.json"
            state = MODULE.snapshot_state(sysfs)
            MODULE.save_state(recovery, state)

            MODULE.apply_request(sysfs, 3_000_000)
            MODULE.verify_request(sysfs, 3_000_000)
            self.assertEqual((sysfs / "cpufreq" / "boost").read_text(), "0\n")

            MODULE.restore_state(state, recovery)
            policy = sysfs / "cpufreq" / "policy0"
            self.assertEqual((policy / "scaling_min_freq").read_text(), "1700000\n")
            self.assertEqual((policy / "scaling_max_freq").read_text(), "5000000\n")
            self.assertEqual((policy / "scaling_governor").read_text(), "powersave\n")
            self.assertTrue(state["restored"])

    def test_parse_turbostat_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "turbostat.txt"
            output.write_text(
                "Busy% Bzy_MHz TSC_MHz PkgTmp\n"
                "75.0 2995 3398 61\n"
                "80.0 3004 3398 62\n"
            )
            self.assertEqual(MODULE.parse_turbostat_mhz(output), [2995.0, 3004.0])

    def test_discovers_common_nominal_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sysfs = self.make_sysfs(Path(directory))
            self.assertEqual(MODULE.nominal_frequency_khz(sysfs), 3_401_000)

    def test_manual_set_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sysfs = self.make_sysfs(root)
            recovery = root / "state.json"
            args = Namespace(
                sysfs_root=sysfs,
                target_khz=None,
                recovery_file=recovery,
            )

            MODULE.set_frequency(args)
            policy = sysfs / "cpufreq" / "policy0"
            self.assertEqual((policy / "scaling_min_freq").read_text(), "3401000\n")
            self.assertEqual((policy / "scaling_max_freq").read_text(), "3401000\n")
            with self.assertRaisesRegex(RuntimeError, "unrestored frequency snapshot"):
                MODULE.set_frequency(args)

            state = MODULE.json.loads(recovery.read_text())
            MODULE.restore_state(state, recovery)
            self.assertEqual((policy / "scaling_min_freq").read_text(), "1700000\n")
            self.assertEqual((policy / "scaling_max_freq").read_text(), "5000000\n")


if __name__ == "__main__":
    unittest.main()
