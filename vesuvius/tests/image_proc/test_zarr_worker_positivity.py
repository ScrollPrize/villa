"""PBS-only actual-module checks for the narrowed0032 worker argument guard."""
import unittest
from unittest import mock

from vesuvius.image_proc.run.zarr_tasks import base


class WorkerArgumentTests(unittest.TestCase):
    def parse(self, value=None):
        args = ["in.zarr", "out.zarr"]
        if value is not None:
            args += ["-n", value]
        return base.get_base_parser().parse_args(args)

    def rejected(self, value):
        with self.assertRaises(SystemExit) as caught:
            self.parse(value)
        self.assertEqual(caught.exception.code, 2)

    def test_zero_workers_rejected(self):
        self.rejected("0")

    def test_negative_workers_rejected(self):
        self.rejected("-1")

    def test_nonnumeric_worker_control(self):
        self.rejected("not-an-int")

    def test_fractional_worker_control(self):
        self.rejected("1.5")

    def test_one_worker_control(self):
        self.assertEqual(base.make_task_config(self.parse("1")).num_workers, 1)

    def test_four_workers_control(self):
        self.assertEqual(base.make_task_config(self.parse("4")).num_workers, 4)

    def supplier_name(self):
        # Compare main and the independently credited PR2295 default mechanism.
        return "available_cpus" if hasattr(base, "available_cpus") else "cpu_count"

    def test_omitted_workers_keeps_supplier_control(self):
        with mock.patch.object(base, self.supplier_name(), return_value=3) as supplier:
            self.assertIsNone(self.parse().num_workers)
            self.assertEqual(base.make_task_config(self.parse()).num_workers, 3)
            supplier.assert_called_once_with()

    def test_positive_workers_do_not_call_supplier_control(self):
        with mock.patch.object(base, self.supplier_name(), side_effect=AssertionError("unexpected default")):
            self.assertEqual(base.make_task_config(self.parse("4")).num_workers, 4)
