"""End-to-end sanity for the runtime-owned interactive session.

Constructs a real InteractiveFitSession (which builds and drives a real
FitContext on the fitter thread) against the local Scroll1 dataset over a
small z-range, waits for Ready, runs two iterations, and checks the pause
protocol outputs (autosave checkpoint + published preview manifest).

Gated behind SPIRAL_INTERACTIVE_E2E=1: it needs a GPU and the local dataset
and takes minutes, so it must run in its own interpreter, not as part of the
default suite:

    SPIRAL_INTERACTIVE_E2E=1 uv run python -m pytest tests/test_interactive_e2e.py -q
"""

import glob
import os
from pathlib import Path
import time
import unittest

DATASET = "/home/paul/projects/vesuvius-scrolls/spiral/dataset"


@unittest.skipUnless(
    os.environ.get("SPIRAL_INTERACTIVE_E2E") == "1",
    "set SPIRAL_INTERACTIVE_E2E=1 (needs a GPU and the local dataset; "
    "run in a dedicated interpreter)")
class InteractiveEndToEndTests(unittest.TestCase):
    def test_session_reaches_ready_runs_and_pauses_with_outputs(self):
        import tempfile

        from fit_session import (SpiralInputPaths, SpiralPreviewConfig,
                                 SpiralRunConfig, load_scroll_spec)
        from spiral_runtime import InteractiveFitSession

        with tempfile.TemporaryDirectory(prefix="spiral-e2e-") as work:
            out_dir = str(Path(work) / "out")
            cache_dir = str(Path(work) / "cache")
            os.makedirs(out_dir)
            os.makedirs(cache_dir)
            paths = SpiralInputPaths.from_mapping({
                "dataset_root": DATASET,
                "umbilicus": f"{DATASET}/umbilicus.json",
                "verified_patches": f"{DATASET}/verified_patches",
                "fibers": f"{DATASET}/fibers",
                "tracks_dbm": f"{DATASET}/tracks/2um_ds2_ps256_surf_v2.dbm",
                "outer_shell": f"{DATASET}/outer_shell",
                "pcls": [
                    {"path": f"{DATASET}/abs_winding.json", "role": "absolute"},
                    {"path": f"{DATASET}/patch-overlap-pcls.json",
                     "role": "patch_overlap"},
                    {"path": f"{DATASET}/relative_windings.json",
                     "role": "relative"},
                    {"path": f"{DATASET}/same_windings.json",
                     "role": "same_winding"},
                    {"path": f"{DATASET}/drawn_control_points.json",
                     "role": "drawn_control_points"},
                ],
                "output_directory": out_dir,
                "cache_directory": cache_dir,
            })
            # The dense (Lasagna) losses are zero-weighted so the session runs
            # without the resident normal/SDT stores, like the golden run.
            run = SpiralRunConfig.from_mapping({
                "z_begin": 10_000,
                "z_end": 11_000,
                "scroll_name": "s1",
                "config": {
                    "dense_spacing_mode": "grad_mag",
                    "loss_weight_dense_spacing_density": 0.0,
                    "loss_weight_dense_normals": 0.0,
                    "loss_weight_dense_spacing": 0.0,
                },
            })
            session = InteractiveFitSession(
                paths, run, SpiralPreviewConfig(), load_scroll_spec(DATASET),
                status_callback=None)
            try:
                deadline = time.monotonic() + 900
                while (session.status()["state"] not in {"Ready", "Error"}
                       and time.monotonic() < deadline):
                    time.sleep(0.5)
                status = session.status()
                self.assertEqual(status["state"], "Ready", status.get("error"))
                self.assertTrue(status["supports_input_incorporation"])
                self.assertEqual(status["current_iteration"], 0)

                session.run(2)
                while (session.status()["state"] not in {"Paused", "Error"}
                       and time.monotonic() < deadline):
                    time.sleep(0.5)
                status = session.status()
                self.assertEqual(status["state"], "Paused", status.get("error"))
                self.assertEqual(status["current_iteration"], 2)
                self.assertIn("total_loss", status["latest_metrics"])

                autosaves = glob.glob(
                    f"{out_dir}/*/checkpoint_autosave.ckpt")
                self.assertEqual(len(autosaves), 1, autosaves)
                manifest = status["preview_manifest_path"]
                self.assertIsNotNone(manifest)
                self.assertTrue(Path(manifest).is_file(), manifest)
                self.assertEqual(status["preview_generation"], 1)

                saved = session.save_checkpoint(
                    str(Path(out_dir) / "explicit.ckpt"), timeout=300.0)
                self.assertTrue(Path(saved).is_file())
            finally:
                session.close(timeout=60.0)


if __name__ == "__main__":
    unittest.main()
