"""`only_flowsnap.json` must carry the `model-init` that `fit.py` needs.

`fit.py` reads `model-init` from the config's merged args (default `seed`) and
passes `init_mode=model_cfg.init_mode` -- i.e. `cylinder_seed` -- to
`optimizer.load_stages_cfg` for that default. `only_flowsnap.json` is a snap-only
re-optimisation of an existing fit: it has no `cyl_init` stage, so with the
default it stops before the first stage with
`ValueError: stages_json: cylinder_seed missing required stage role(s): ['cyl_init']`.

`optimizer.load_stages_cfg` alone does not show this, because with no `model-init`
key and no `init-mode` key it passes `init_mode=None` and skips the role check;
the failure only appears on the path `fit.py` takes. The test below mirrors that
path instead of calling the loader with its defaults.

Both files set `model-init` to `model`, as do the other snap/offset configs that
re-optimise an existing fit (`offset.json`, `atlas_reopt_snap_only.json`,
`atlas_reopt_snap_flow_speculative.json`, `corr_snap.json`,
`reopt_snap_surf_self_d.json`); the `init_*` configs set `seed` because they
build a model from a seed.

The flow-gate flag placement that #2458 was first reported for is #2463's change
(`tifxyz-flow-gate-channels` in `args`, not in `base`) and is covered by
`test_approval_inpaint_configs.py`; this file guards the part that #2463 leaves:
`only_flowsnap.json` loading on the `fit.py` path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

CONFIG_DIR = Path(ROOT) / "configs"

FLOW_SNAP_CONFIGS = ("only_flowsnap.json", "only_flowsnap_speculative.json")


def _importable(name: str) -> bool:
	if name in sys.modules:
		return True
	try:
		return importlib.util.find_spec(name) is not None
	except (ImportError, ValueError):
		return False


# `optimizer` imports torch and, through `cli_data` -> `fit_data`, the compiled
# `vc_reader.reader` extension, so the load test below only runs where a real villa
# install has both, and a checkout without them skips it instead of erroring --
# the same handling `test_approval_inpaint_configs.py` gives that import. The probe
# names `vc_reader.reader` rather than `vc_reader`: from the repository root a
# checkout's unbuilt `vc_reader/` directory resolves as a namespace package, so a
# probe of `vc_reader` alone would let the import fail inside the test.
HAS_OPTIMIZER_DEPS = all(_importable(name) for name in ("torch", "numpy", "vc_reader.reader"))


def _fit_init_mode(args: dict) -> str | None:
	"""The `init_mode` `fit.py` passes to `load_stages_cfg` for this config's args."""
	model_init = str(args.get("model-init", "seed")).strip().lower()
	return "cylinder_seed" if model_init == "seed" else None


class FlowSnapConfigTest(unittest.TestCase):
	def _read(self, name: str) -> dict:
		return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))

	def test_flow_snap_configs_set_model_init_to_model(self) -> None:
		for name in FLOW_SNAP_CONFIGS:
			with self.subTest(config=name):
				args = self._read(name).get("args") or {}
				self.assertEqual(str(args.get("model-init", "seed")).strip().lower(), "model")
				# A snap-only config runs on fit.py's `init_mode=None` path.
				self.assertIsNone(_fit_init_mode(args))

	@unittest.skipUnless(
		HAS_OPTIMIZER_DEPS,
		"needs torch and the compiled vc_reader extension to import optimizer",
	)
	def test_flow_snap_configs_load_the_way_fit_does(self) -> None:
		try:
			import optimizer
		except ImportError as exc:  # a full villa install is not there after all
			self.skipTest(f"importing optimizer needs a full villa install: {exc}")

		for name in FLOW_SNAP_CONFIGS:
			with self.subTest(config=name):
				cfg = self._read(name)
				args = dict(cfg.get("args") or {})
				# fit.py merges `args` into the CLI namespace and drops the block
				# before loading the stages.
				init_mode = _fit_init_mode(args)
				cfg.pop("args", None)
				self.assertTrue(optimizer.load_stages_cfg(cfg, init_mode=init_mode))


if __name__ == "__main__":
	unittest.main()
