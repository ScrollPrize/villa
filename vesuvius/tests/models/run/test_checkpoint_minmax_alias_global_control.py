"""Prepared PBS-only CPU checkpoint/store controls; no network inference."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr

from vesuvius.data.vc_dataset import VCDataset
from vesuvius.models.run import patch_writer
from vesuvius.models.training.normalization import get_normalization
from vesuvius.models.training.save_checkpoint import save_checkpoint


CASES = {
    "current_minmax": ("current", "minmax", "instance_minmax"),
    "current_rescale": ("current", "rescale_to_01", "instance_minmax"),
    "legacy_minmax": ("legacy", "minmax", "instance_minmax"),
    "legacy_rescale": ("legacy", "rescale_to_01", "instance_minmax"),
    "none_control": ("current", "none", "none"),
    "zscore_control": ("current", "zscore", "global_zscore"),
    "nnunet_rescale_control": ("nnunet", "RescaleTo01Normalization", "instance_minmax"),
    "cli_control": ("missing", None, "instance_minmax"),
}


def source_bindings():
    path = Path(patch_writer.__file__).with_name("inference.py")
    tree = ast.parse(path.read_text(encoding="utf8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Inferer")
    names = {"_create_dataset_and_loader", "_limit_dataset_patches"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == 2
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
        "_checkpoint_normalization_scheme", "_nnunet_normalization_from_model_info"}]
    mapping = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "_NNUNET_NORMALIZATION_MAP" for t in n.targets))
    namespace = dict(torch=torch, np=np, VCDataset=VCDataset, DataLoader=DataLoader)
    exec(compile(ast.Module(body=[mapping] + helpers + methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class CheckpointMinmaxAliasTests(unittest.TestCase):
    def run_case(self, name):
        header_kind, scheme, expected_scheme = CASES[name]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
            if name == "current_rescale": values.fill(7)
            source = root / "input.zarr"
            array = zarr.open_array(str(source), mode="w", shape=values.shape,
                                    chunks=values.shape, dtype=values.dtype)
            array[:] = values
            before = tree_hashes(source)
            bindings = source_bindings()
            properties = {"mean": 7.0, "std": 3.0} if name == "zscore_control" else {}
            checkpoint_path = root / "fixture.pth"
            if header_kind == "current":
                model = torch.nn.Linear(1, 1, bias=False, device="cpu")
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
                save_checkpoint(model, optimizer, scheduler, 0, str(checkpoint_path),
                    model_config={"supplied_fixture": True},
                    train_dataset=SimpleNamespace(normalization_scheme=scheme,
                                                  intensity_properties=properties))
            elif header_kind == "legacy":
                torch.save({"config": {"image_normalization": scheme}}, checkpoint_path)
            else:
                torch.save({}, checkpoint_path)
            header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_scheme = bindings["_checkpoint_normalization_scheme"](header)
            if header_kind == "nnunet":
                model_scheme, properties = bindings["_nnunet_normalization_from_model_info"]({
                    "configuration_manager": SimpleNamespace(normalization_schemes=[scheme],
                                                              use_mask_for_norm=[False]),
                    "plans_manager": SimpleNamespace()})
            state = SimpleNamespace(input=str(source), patch_size=values.shape, overlap=0.5,
                num_parts=1, part_id=0, model_normalization_scheme=model_scheme,
                normalization_scheme="instance_minmax" if header_kind == "missing" else "instance_zscore",
                model_intensity_properties=properties or None, input_format="zarr", verbose=False,
                skip_empty_patches=False, scroll_id=None, segment_id=None, energy=None,
                resolution=None, input_anon=False, bbox=None, read_retries=1, chunk_cache_mb=0,
                device=torch.device("cpu"), max_patches=None, batch_size=1, num_dataloader_workers=0)
            state._limit_dataset_patches = MethodType(bindings["_limit_dataset_patches"], state)
            try:
                dataset, loader = bindings["_create_dataset_and_loader"](state)
            except ValueError as error:
                if "Invalid normalization_scheme" not in str(error): raise
                self.fail(f"Saved supported normalization {scheme!r} was rejected: {error}")
            self.assertEqual(dataset.volume.normalization_scheme, expected_scheme)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(state.num_active_patches, 1)
            training_scheme = scheme if header_kind in {"current", "legacy"} else "minmax"
            if name == "zscore_control":
                # Saved statistics select global inference normalization, not per-instance training zscore.
                self.assertEqual(header["intensity_properties"], properties)
                self.assertEqual(dataset.volume.global_mean, 7.0)
                self.assertEqual(dataset.volume.global_std, 3.0)
                expected = (values.astype(np.float32) - np.float32(7.0)) / np.float32(3.0)
                instance_expected = get_normalization("zscore", properties).run(values)
                self.assertFalse(np.array_equal(expected, instance_expected))
            else:
                normalizer = get_normalization(training_scheme, properties)
                expected = values.astype(np.float32) if normalizer is None else normalizer.run(values)
            np.testing.assert_array_equal(dataset[0]["data"].numpy()[0], expected)
            batch = next(iter(loader))
            np.testing.assert_array_equal(batch["data"].numpy()[0, 0], expected)
            self.assertEqual(batch["pos"], [(0, 0, 0)])
            self.assertEqual(tree_hashes(source), before)
            if header_kind == "current":
                self.assertEqual(header["normalization_scheme"], scheme)

    def test_current_minmax(self): self.run_case("current_minmax")
    def test_current_rescale(self): self.run_case("current_rescale")
    def test_legacy_minmax(self): self.run_case("legacy_minmax")
    def test_legacy_rescale(self): self.run_case("legacy_rescale")
    def test_none_control(self): self.run_case("none_control")
    def test_zscore_control(self): self.run_case("zscore_control")
    def test_nnunet_rescale_control(self): self.run_case("nnunet_rescale_control")
    def test_cli_control(self): self.run_case("cli_control")
