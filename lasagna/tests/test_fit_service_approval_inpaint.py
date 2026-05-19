from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import fit_service


def _b64(data: bytes) -> str:
	return base64.b64encode(data).decode("ascii")


class FitServiceApprovalInpaintTest(unittest.TestCase):
	def test_seed_mode_accepts_generic_tifxyz_and_ignores_without_consumer(self) -> None:
		body = {
			"tifxyz": {
				"x.tif": _b64(b"x"),
				"y.tif": _b64(b"y"),
				"z.tif": _b64(b"z"),
				"meta.json": _b64(b"{}"),
				"approval.tif": _b64(b"approval"),
				"d.tif": _b64(b"d"),
			}
		}
		cfg = {"args": {"model-init": "seed"}}
		args = cfg["args"]

		with tempfile.TemporaryDirectory() as td:
			tifxyz_dir = fit_service._decode_tifxyz_for_request(
				body=body,
				cfg=cfg,
				args_section=args,
				tmp_dir=td,
				model_init="seed",
				ext_offset_enabled=False,
			)

			self.assertIsNotNone(tifxyz_dir)
			self.assertEqual((Path(tifxyz_dir) / "approval.tif").read_bytes(), b"approval")
			self.assertNotIn("approval-inpaint-tifxyz", args)
			self.assertNotIn("tifxyz-init", args)

	def test_approval_inpaint_wires_generic_tifxyz(self) -> None:
		body = {"tifxyz": {"x.tif": _b64(b"x"), "approval.tif": _b64(b"approval")}}
		cfg = {"args": {"model-init": "seed", "approval-inpaint": True}}
		args = cfg["args"]

		with tempfile.TemporaryDirectory() as td:
			tifxyz_dir = fit_service._decode_tifxyz_for_request(
				body=body,
				cfg=cfg,
				args_section=args,
				tmp_dir=td,
				model_init="seed",
				ext_offset_enabled=False,
			)

			self.assertIsNotNone(tifxyz_dir)
			self.assertEqual(args["approval-inpaint-tifxyz"], tifxyz_dir)
			self.assertNotIn("tifxyz-init", args)

	def test_approval_inpaint_requires_seed_model_init(self) -> None:
		body = {"tifxyz": {"x.tif": _b64(b"x")}}
		cfg = {"args": {"model-init": "ext", "approval-inpaint": True}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "only valid with args.model-init=seed"):
				fit_service._decode_tifxyz_for_request(
					body=body,
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="ext",
					ext_offset_enabled=False,
				)

	def test_approval_inpaint_requires_generic_tifxyz(self) -> None:
		cfg = {"args": {"model-init": "seed", "approval-inpaint": True}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "approval-inpaint requires request tifxyz"):
				fit_service._decode_tifxyz_for_request(
					body={},
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="seed",
					ext_offset_enabled=False,
				)

	def test_ext_init_requires_generic_tifxyz(self) -> None:
		cfg = {"args": {"model-init": "ext"}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "model-init=ext requires request tifxyz"):
				fit_service._decode_tifxyz_for_request(
					body={},
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="ext",
					ext_offset_enabled=False,
				)

	def test_ext_offset_requires_generic_tifxyz(self) -> None:
		cfg = {"args": {"model-init": "seed"}}

		with tempfile.TemporaryDirectory() as td:
			with self.assertRaisesRegex(ValueError, "ext_offset is enabled but request has no tifxyz"):
				fit_service._decode_tifxyz_for_request(
					body={},
					cfg=cfg,
					args_section=cfg["args"],
					tmp_dir=td,
					model_init="seed",
					ext_offset_enabled=True,
				)

	def test_seed_mode_ignores_surplus_model_transport(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			model_input = fit_service._decode_model_for_request(
				body={"model_data": _b64(b"checkpoint")},
				tmp_dir=td,
				model_init="seed",
			)

			self.assertIsNone(model_input)
			self.assertFalse((Path(td) / "model_input.pt").exists())

	def test_model_mode_decodes_model_transport(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			model_input = fit_service._decode_model_for_request(
				body={"model_data": _b64(b"checkpoint")},
				tmp_dir=td,
				model_init="model",
			)

			self.assertIsNotNone(model_input)
			self.assertEqual(Path(model_input).read_bytes(), b"checkpoint")

	def test_ext_mode_applies_window_transport(self) -> None:
		args: dict[str, object] = {"model-init": "ext"}

		fit_service._apply_window_transport_args(
			body={"window_size": 4096, "window_overlap": 256},
			args_section=args,
			model_init="ext",
		)

		self.assertEqual(args["window-size"], 4096)
		self.assertEqual(args["window-overlap"], 256)

	def test_model_mode_ignores_window_transport(self) -> None:
		args: dict[str, object] = {"model-init": "model"}

		fit_service._apply_window_transport_args(
			body={"window_size": 4096, "window_overlap": 256},
			args_section=args,
			model_init="model",
		)

		self.assertNotIn("window-size", args)
		self.assertNotIn("window-overlap", args)


if __name__ == "__main__":
	unittest.main()
