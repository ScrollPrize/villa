"""Tests for the preview self-intersection publication policy.

The census tool is faked with small executables so every state the policy
distinguishes -- clean, dirty, tool failure, tool absent -- is exercised
without a compiled vc_tifxyz_selfcross or a Lasagna run. The one contract
these tests pin above all others: ``off`` touches nothing, and a rejection
leaves the previously published preview exactly as it was.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spiral_service import (PreviewSelfcrossRejected,
                            _apply_preview_selfcross_policy,
                            _census_flattened_preview,
                            _find_selfcross_tool)

CLEAN_REPORT = {
    "clean_of_transverse_self_intersection": True,
    "census": [
        {"diagonal": 0, "transverse": 0},
        {"diagonal": 1, "transverse": 0},
    ],
}
DIRTY_REPORT = {
    "clean_of_transverse_self_intersection": False,
    "census": [
        {"diagonal": 0, "transverse": 12},
        {"diagonal": 1, "transverse": 9},
    ],
}


def _tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


class SelfcrossFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="spiral_selfcross_test_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.surface = self.root / "surface-lasagna.tifxyz"
        self.surface.mkdir()
        (self.surface / "meta.json").write_text("{}")
        self.publish_parent = self.root / "published"
        self.publish_parent.mkdir()
        self.publish_root = self.publish_parent / ".generation-2.incoming"
        self.publish_root.mkdir()
        (self.publish_root / "loss.png").write_bytes(b"loss")

    def fake_tool(self, report=None, exit_code=0, with_collection=True):
        """An executable standing in for vc_tifxyz_selfcross."""
        tool = self.root / "fake_selfcross"
        payload = json.dumps(report or {})
        lines = ["#!/bin/sh"]
        if report is not None:
            lines.append(f"cat > \"$3\" <<'JSON'\n{payload}\nJSON")
        if with_collection and report is not None:
            lines.append('printf \'{"collections": {}}\' > "$5"')
        lines.append(f"exit {exit_code}")
        tool.write_text("\n".join(lines) + "\n")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
        return tool

    def apply(self, policy, tool, generation=2, published=None):
        published = published if published is not None else {}
        with mock.patch.dict(os.environ,
                             {"VC_SELFCROSS_PATH": str(tool)} if tool
                             else {}, clear=False):
            if tool is None:
                os.environ.pop("VC_SELFCROSS_PATH", None)
            _apply_preview_selfcross_policy(
                policy, self.surface, self.publish_root,
                self.publish_parent, generation, published)
        return published


class CleanPreviewPublishes(SelfcrossFixture):
    def test_reject_mode_passes_a_clean_census_through(self):
        tool = self.fake_tool(CLEAN_REPORT)
        published = self.apply("reject", tool)
        entry = published["selfcross"]
        self.assertEqual(entry["state"], "clean")
        self.assertEqual(entry["policy"], "reject")
        self.assertEqual(entry["transverse"], {"d0": 0, "d1": 0})
        self.assertEqual(entry["report"], "selfcross-report.json")
        self.assertEqual(entry["collection"], "selfcross-sites.json")
        self.assertEqual(entry["tool"], "fake_selfcross")
        # both artifacts live inside the generation being published
        self.assertTrue((self.publish_root / "selfcross-report.json").is_file())
        self.assertTrue((self.publish_root / "selfcross-sites.json").is_file())
        # nothing was withheld: no rejection evidence appeared
        self.assertEqual(
            list(self.publish_parent.glob("*.selfcross-rejection-*.json")), [])


class CrossingPreviewIsWithheld(SelfcrossFixture):
    def test_reject_mode_raises_before_promotion_with_evidence(self):
        tool = self.fake_tool(DIRTY_REPORT)
        with self.assertRaises(PreviewSelfcrossRejected):
            self.apply("reject", tool)
        candidates = sorted(self.publish_parent.glob(
            "generation-2.selfcross-rejection-*.json"))
        self.assertEqual(len(candidates), 1,
                         "rejection evidence must survive staging cleanup")
        evidence = json.loads(candidates[0].read_text())
        self.assertEqual(evidence["state"], "dirty")
        self.assertEqual(evidence["generation"], 2)
        self.assertEqual(evidence["transverse"], {"d0": 12, "d1": 9})
        self.assertFalse(
            evidence["report_json"]["clean_of_transverse_self_intersection"])


class ReportModePublishesWithDiagnostics(SelfcrossFixture):
    def test_dirty_census_is_recorded_but_never_blocks(self):
        tool = self.fake_tool(DIRTY_REPORT)
        published = self.apply("report", tool)   # must not raise
        entry = published["selfcross"]
        self.assertEqual(entry["state"], "dirty")
        self.assertEqual(entry["transverse"], {"d0": 12, "d1": 9})
        self.assertTrue((self.publish_root / "selfcross-report.json").is_file())
        self.assertEqual(
            list(self.publish_parent.glob("*.selfcross-rejection-*.json")), [])

    def test_tool_failure_is_recorded_but_never_blocks(self):
        tool = self.fake_tool(report=None, exit_code=1)
        published = self.apply("report", tool)   # must not raise
        self.assertEqual(published["selfcross"]["state"], "error")


class ToolFailureDoesNotPublish(SelfcrossFixture):
    def test_reject_mode_fails_closed_on_tool_error(self):
        tool = self.fake_tool(report=None, exit_code=1)
        with self.assertRaises(PreviewSelfcrossRejected):
            self.apply("reject", tool)
        evidence = json.loads(next(self.publish_parent.glob(
            "generation-2.selfcross-rejection-*.json")).read_text())
        self.assertEqual(evidence["state"], "error")

    def test_reject_mode_fails_closed_when_tool_is_absent(self):
        with mock.patch("spiral_service.shutil.which", return_value=None):
            with self.assertRaises(PreviewSelfcrossRejected):
                self.apply("reject", None)
        evidence = json.loads(next(self.publish_parent.glob(
            "generation-2.selfcross-rejection-*.json")).read_text())
        self.assertEqual(evidence["state"], "not_run")

    def test_unreadable_report_is_an_error(self):
        tool = self.fake_tool({"unexpected": "shape"})
        summary = _census_flattened_preview(
            self.surface, self.publish_root, tool=tool)
        self.assertEqual(summary["state"], "error")
        self.assertIn("invalid selfcross report", summary["detail"])


class ReportSchemaIsValidatedNotTrusted(SelfcrossFixture):
    """R49: cleanliness derives from validated counts; the tool's own
    flag must agree with them, and partial diagnostics never ship."""

    MISSING_DIAGONAL = {
        "clean_of_transverse_self_intersection": True,
        "census": [{"diagonal": 0, "transverse": 0}],
    }
    LYING_CLEAN_FLAG = {
        "clean_of_transverse_self_intersection": True,
        "census": [
            {"diagonal": 0, "transverse": 3},
            {"diagonal": 1, "transverse": 0},
        ],
    }

    def test_missing_diagonal_is_an_error_and_fails_closed(self):
        tool = self.fake_tool(self.MISSING_DIAGONAL)
        summary = _census_flattened_preview(
            self.surface, self.publish_root, tool=tool)
        self.assertEqual(summary["state"], "error")
        with self.assertRaises(PreviewSelfcrossRejected):
            self.apply("reject", tool)

    def test_clean_flag_contradicting_counts_is_an_error_and_fails_closed(
            self):
        tool = self.fake_tool(self.LYING_CLEAN_FLAG)
        summary = _census_flattened_preview(
            self.surface, self.publish_root, tool=tool)
        self.assertEqual(summary["state"], "error")
        self.assertIn("contradicts", summary["detail"])
        with self.assertRaises(PreviewSelfcrossRejected):
            self.apply("reject", tool)

    def test_failed_census_leaves_no_partial_diagnostics(self):
        # a valid report written, then the tool dies (e.g. while writing
        # the point collection): nothing may remain in the generation
        tool = self.fake_tool(CLEAN_REPORT, exit_code=1)
        summary = _census_flattened_preview(
            self.surface, self.publish_root, tool=tool)
        self.assertEqual(summary["state"], "error")
        self.assertFalse(
            (self.publish_root / "selfcross-report.json").exists())
        self.assertFalse(
            (self.publish_root / "selfcross-sites.json").exists())

    def test_invalid_report_is_also_cleaned_up(self):
        tool = self.fake_tool(self.LYING_CLEAN_FLAG)
        _census_flattened_preview(self.surface, self.publish_root, tool=tool)
        self.assertFalse(
            (self.publish_root / "selfcross-report.json").exists())
        self.assertFalse(
            (self.publish_root / "selfcross-sites.json").exists())


class OffIsByteIdentical(SelfcrossFixture):
    def test_off_touches_neither_staging_nor_manifest_nor_environment(self):
        # a tool that would explode if ever invoked (created BEFORE the
        # baseline digest so only the policy could change the tree)
        bomb = self.fake_tool(report=None, exit_code=99)
        before = _tree_digest(self.root)
        published = {"surface_id": "s", "schema_version": 3}
        manifest_before = dict(published)
        result = _apply_preview_selfcross_policy(
            "off", self.surface, self.publish_root,
            self.publish_parent, 2, published)
        self.assertIsNone(result)
        self.assertEqual(published, manifest_before)
        self.assertEqual(_tree_digest(self.root), before,
                         "'off' must leave every byte untouched")
        self.assertNotIn("selfcross", published)
        del bomb


class EarlierPreviewSurvivesRejection(SelfcrossFixture):
    def test_generation_1_is_untouched_when_generation_2_is_rejected(self):
        # generation-1 was published earlier and must not change
        final_1 = self.publish_parent / "generation-1"
        final_1.mkdir()
        (final_1 / "manifest.json").write_text(
            json.dumps({"surface_id": "gen1"}))
        (final_1 / "surface.bin").write_bytes(b"gen1-bytes")
        gen1_before = _tree_digest(final_1)

        tool = self.fake_tool(DIRTY_REPORT)
        with self.assertRaises(PreviewSelfcrossRejected):
            self.apply("reject", tool)
        # mimic the service's failure path: staging is discarded
        shutil.rmtree(self.publish_root, ignore_errors=True)

        self.assertEqual(_tree_digest(final_1), gen1_before,
                         "a rejection must not disturb the published preview")
        self.assertFalse((self.publish_parent / "generation-2").exists(),
                         "the rejected generation must never be promoted")
        self.assertEqual(len(list(self.publish_parent.glob(
            "generation-2.selfcross-rejection-*.json"))), 1)


class ToolDiscovery(unittest.TestCase):
    def test_set_but_missing_override_is_an_error_not_a_fallback(self):
        with mock.patch.dict(os.environ,
                             {"VC_SELFCROSS_PATH": "/nonexistent/tool"}):
            with self.assertRaises(RuntimeError):
                _find_selfcross_tool()

    def test_absent_tool_resolves_to_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VC_SELFCROSS_PATH", None)
            with mock.patch("spiral_service.shutil.which",
                            return_value=None):
                self.assertIsNone(_find_selfcross_tool())


if __name__ == "__main__":
    unittest.main()
