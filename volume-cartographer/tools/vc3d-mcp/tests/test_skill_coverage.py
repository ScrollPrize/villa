from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from vc3d_mcp import core
from vc3d_mcp import tools as _tools  # noqa: F401 - registers MCP tools


VC_ROOT = Path(__file__).resolve().parents[3]
AUTHORING_SKILLS_ROOT = VC_ROOT / ".claude" / "skills"
SKILLS_ROOT = VC_ROOT / ".agents" / "skills"
DESCRIPTION_PATH = VC_ROOT / "apps/VC3D/agent_bridge/rpc_description.json"
CORE_LINE_LIMIT = 250

# Route contract namespaces to workflow skills. This intentionally describes
# workflows, not a duplicated list of tool schemas.
NAMESPACE_SKILLS = {
    "ping": "vc3d-bridge-session",
    "state": "vc3d-bridge-session",
    "workspace": "vc3d-bridge-session",
    "job": "vc3d-bridge-session",
    "project": "vc3d-open-data",
    "volume": "vc3d-open-data",
    "catalog": "vc3d-open-data",
    "viewer": "vc3d-visual-evidence",
    "canvas": "vc3d-visual-evidence",
    "screenshot": "vc3d-visual-evidence",
    "segments": "vc3d-segment-lifecycle",
    "segment": "vc3d-segment-lifecycle",
    "tags": "vc3d-segment-lifecycle",
    "segmentation": "vc3d-segmentation-editing",
    "tracer": "vc3d-segmentation-editing",
    "points": "vc3d-winding-annotation",
    "wrap_annotation": "vc3d-winding-annotation",
    "seeding": "vc3d-seeding",
    "fiber": "vc3d-fiber-tracing",
    "render": "vc3d-rendering",
    "flatten": "vc3d-flattening",
    "lasagna": "vc3d-lasagna",
    "atlas": "vc3d-atlas",
    "spiral": "vc3d-spiral",
}
MCP_ONLY_TOOL_SKILLS = {"vc3d_wait_job": "vc3d-bridge-session"}
EXPECTED_BRIDGE_ONLY_METHODS = {"rpc.describe"}
STANDALONE_WORKFLOW_SKILLS = {
    "scripts/spiral/flatten_spiral_checkpoint.py":
        "vc3d-spiral-checkpoint-flattening",
}


def _frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise AssertionError(f"{path}: missing opening frontmatter delimiter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise AssertionError(f"{path}: missing closing frontmatter delimiter") from exc

    values: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            raise AssertionError(f"{path}: unsupported frontmatter line {line!r}")
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


class SkillCoverageTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.description = json.loads(DESCRIPTION_PATH.read_text(encoding="utf-8"))

    async def test_every_registered_tool_has_a_primary_workflow_skill(self) -> None:
        registered = {tool.name for tool in await core.mcp.list_tools()}
        routed: dict[str, str] = {}
        bridge_only: set[str] = set()

        for method, contract in self.description["methods"].items():
            mcp = contract.get("mcp")
            if mcp is None:
                bridge_only.add(method)
                continue
            namespace = method.split(".", 1)[0]
            self.assertIn(namespace, NAMESPACE_SKILLS, method)
            routed[mcp["tool"]] = NAMESPACE_SKILLS[namespace]

        routed.update(MCP_ONLY_TOOL_SKILLS)
        self.assertEqual(bridge_only, EXPECTED_BRIDGE_ONLY_METHODS)
        self.assertEqual(set(routed), registered)
        for tool, skill in routed.items():
            with self.subTest(tool=tool, skill=skill):
                self.assertTrue((SKILLS_ROOT / skill / "SKILL.md").is_file())

    def test_codex_discovery_mirrors_authoring_skills(self) -> None:
        authored = {
            path.parent.name
            for path in AUTHORING_SKILLS_ROOT.glob("*/SKILL.md")
        }
        discovered = {
            path.parent.name
            for path in SKILLS_ROOT.glob("*/SKILL.md")
        }
        self.assertGreater(len(authored), 0)
        self.assertEqual(discovered, authored)

        for name in sorted(authored):
            with self.subTest(skill=name):
                discovered_dir = SKILLS_ROOT / name
                self.assertTrue(discovered_dir.is_symlink())
                self.assertTrue(discovered_dir.samefile(AUTHORING_SKILLS_ROOT / name))

    def test_skill_structure_is_compact_and_resolvable(self) -> None:
        skill_files = sorted(SKILLS_ROOT.glob("*/SKILL.md"))
        self.assertGreater(len(skill_files), 0)

        for path in skill_files:
            with self.subTest(skill=path.parent.name):
                text = path.read_text(encoding="utf-8")
                metadata = _frontmatter(path)
                self.assertEqual(set(metadata), {"name", "description"})
                self.assertEqual(metadata["name"], path.parent.name)
                self.assertTrue(metadata["description"])
                self.assertLessEqual(len(text.splitlines()), CORE_LINE_LIMIT)

                for relative in re.findall(r"\]\((references/[^)#]+)(?:#[^)]+)?\)", text):
                    self.assertTrue((path.parent / relative).is_file(), relative)

    def test_standalone_workflows_have_primary_skills(self) -> None:
        for relative, skill in STANDALONE_WORKFLOW_SKILLS.items():
            with self.subTest(workflow=relative, skill=skill):
                source = VC_ROOT / relative
                skill_path = SKILLS_ROOT / skill / "SKILL.md"
                self.assertTrue(source.is_file(), relative)
                self.assertTrue(skill_path.is_file(), skill)
                self.assertIn(source.name, skill_path.read_text(encoding="utf-8"))

    def test_long_references_have_contents(self) -> None:
        for path in sorted(SKILLS_ROOT.glob("*/references/*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) <= 100:
                continue
            with self.subTest(reference=path.relative_to(SKILLS_ROOT)):
                self.assertIn("## Contents", lines[:40])


if __name__ == "__main__":
    unittest.main()
