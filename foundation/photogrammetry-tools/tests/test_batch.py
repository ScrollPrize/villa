"""
Checks for the batch helpers (prepare_input.py, batch/linked.py, batch/scale_scene.py, batch/split_gate.py).

Run with:  python -m unittest discover -s tests   (from foundation/photogrammetry-tools)
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = os.path.join(HERE, "batch")


def run(script, *args):
    return subprocess.run([sys.executable, script, *args], capture_output=True, text=True)


def view(key, name, pose):
    return {"key": key, "value": {"ptr_wrapper": {"data": {"filename": name, "id_pose": pose}}}}


ALIGN_LOG = """best: axes {best} shift +1, 16265 rim pairs, rms {rms} mm
ranking: axes {best} shift +1 score {s1}; axes {best} shift +0 score {s1}; axes {other} shift +0 score {s2}; axes {other} shift +1 score {s2}
scale of B relative to A that best fits the rim: {rim} (not applied; 1 = the two marker scales agree on the object's size)
"""


class SplitGateTest(unittest.TestCase):
    def gate(self, opt, s1=2.9, s2=0.9, rms=0.059, rim=1.0024):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "align.log"), "w").write(
                ALIGN_LOG.format(best="+1-1", other="+1+1", s1=s1, s2=s2, rms=rms, rim=rim))
            open(os.path.join(d, "silhouette.log"), "w").write(
                f"outline error is smallest at a shift of {opt:+.3f} mm along the thin axis\n")
            r = run(os.path.join(BATCH, "split_gate.py"), d)
            return r.returncode, json.load(open(os.path.join(d, "auto_decision.json")))

    def test_decisions(self):
        code, res = self.gate(+0.145)                       # 1585: within 0.2 mm, no shift
        self.assertEqual((code, res["thin_shift_mm"], res["review_unshifted_variant"]), (0, 0.0, False))
        code, res = self.gate(+0.305)                       # 1482: shifted, flagged for the two-variant review
        self.assertEqual((code, res["thin_shift_mm"], res["review_unshifted_variant"]), (0, 0.3, True))
        code, res = self.gate(+1.0)                         # 1756: thicker, reviewed
        self.assertEqual((code, res["thin_shift_mm"]), (0, 1.0))
        self.assertEqual(self.gate(+0.914, s1=2.2, s2=1.5)[0], 1)   # sncass94 before regrouping: no clear flip
        self.assertEqual(self.gate(+1.6)[0], 1)                     # optimum outside the sweep
        self.assertEqual(self.gate(0.0, rim=1.02)[0], 1)            # marker scales disagree


class LinkedTest(unittest.TestCase):
    def test_partly_posed_orientation_is_not_linked(self):
        with tempfile.TemporaryDirectory() as d:
            lists = os.path.join(d, "lists")
            os.makedirs(lists)
            open(os.path.join(lists, "keep-o1-all.txt"), "w").write("a.jpg\nb.jpg\n")
            open(os.path.join(lists, "keep-o2-all.txt"), "w").write("c.jpg\nd.jpg\n")
            scene = {"extrinsics": [{"key": 0}, {"key": 1}, {"key": 2}],
                     "views": [view(0, "a.jpg", 0), view(1, "b.jpg", 1), view(2, "c.jpg", 2), view(3, "d.jpg", 9)]}
            json.dump(scene, open(os.path.join(d, "s.json"), "w"))
            r = run(os.path.join(BATCH, "linked.py"), os.path.join(d, "s.json"), lists)
            self.assertEqual(r.returncode, 1)
            self.assertIn("o2: 1/2 posed (50%)", r.stdout)
            self.assertEqual(run(os.path.join(BATCH, "linked.py"), os.path.join(d, "s.json"), lists,
                                 "--min-frac", "0.5").returncode, 0)


class ScaleSceneTest(unittest.TestCase):
    def test_centres_and_landmarks_are_scaled(self):
        with tempfile.TemporaryDirectory() as d:
            scene = {"extrinsics": [{"key": 0, "value": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "center": [1, 2, 3]}}],
                     "structure": [{"key": 0, "value": {"X": [0.5, 0, -1], "observations": []}}], "control_points": []}
            json.dump(scene, open(os.path.join(d, "in.json"), "w"))
            run(os.path.join(BATCH, "scale_scene.py"), os.path.join(d, "in.json"), os.path.join(d, "out.json"), "2.5")
            out = json.load(open(os.path.join(d, "out.json")))
            self.assertEqual(out["extrinsics"][0]["value"]["center"], [2.5, 5.0, 7.5])
            self.assertEqual(out["extrinsics"][0]["value"]["rotation"], scene["extrinsics"][0]["value"]["rotation"])
            self.assertEqual(out["structure"][0]["value"]["X"], [1.25, 0.0, -2.5])


class PrepareInputTest(unittest.TestCase):
    def test_inputs_lists_and_split_inputs(self):
        with tempfile.TemporaryDirectory() as work:
            tree = os.path.join(work, "0608", "0608")
            for b, stems in (("orientation1_camera_low", ["DSC1", "DSC2"]), ("orientation2_camera_high", ["DSC3"])):
                for sub in ("JPGEnhanced", "ScrollMasks"):
                    os.makedirs(os.path.join(tree, b, sub))
                for s in stems:
                    open(os.path.join(tree, b, "JPGEnhanced", s + ".jpg"), "w").write(s)
                    open(os.path.join(tree, b, "ScrollMasks", s + "_mask.png"), "w").write(s)
            r = run(os.path.join(HERE, "prepare_input.py"), work, "0608")
            self.assertEqual(r.returncode, 0, r.stderr)
            base = os.path.join(work, "0608")
            self.assertEqual(sorted(os.listdir(os.path.join(base, "pgs-input"))),
                             ["DSC1.jpg", "DSC1_mask.png", "DSC2.jpg", "DSC2_mask.png", "DSC3.jpg", "DSC3_mask.png"])
            self.assertEqual(open(os.path.join(base, "recon", "scale-lists", "keep-o1-low.txt")).read(), "DSC1.jpg\nDSC2.jpg\n")
            self.assertEqual(open(os.path.join(base, "split", "run-record", "keep-o2-all.txt")).read(), "DSC3.jpg\n")
            self.assertEqual(sorted(os.listdir(os.path.join(base, "split", "pgs-input-o2"))), ["DSC3.jpg", "DSC3_mask.png"])
            # a photograph without its mask stops the preparation
            os.remove(os.path.join(tree, "orientation1_camera_low", "ScrollMasks", "DSC2_mask.png"))
            self.assertNotEqual(run(os.path.join(HERE, "prepare_input.py"), work, "0608").returncode, 0)


if __name__ == "__main__":
    unittest.main()
