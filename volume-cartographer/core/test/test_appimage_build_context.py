import unittest
from pathlib import Path


VC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = VC_ROOT.parent


class AppImageBuildContextTests(unittest.TestCase):
    def test_docker_context_preserves_runtime_service_siblings(self):
        dockerfile = (VC_ROOT / "scripts" / "Dockerfile.appimage").read_text()
        workflow = (REPO_ROOT / ".github" / "workflows" / "vc3d-linux.yml").read_text()

        self.assertIn("COPY spiral-fitting/ /src/spiral-fitting/", dockerfile)
        self.assertIn(
            "COPY vesuvius/src/vc3d_fiber_format/ "
            "/src/vesuvius/src/vc3d_fiber_format/",
            dockerfile,
        )
        self.assertIn("WORKDIR /src/volume-cartographer", dockerfile)
        self.assertIn("-f scripts/Dockerfile.appimage", workflow)
        self.assertRegex(workflow, r"-f scripts/Dockerfile\.appimage \\\n\s+\.\.")


if __name__ == "__main__":
    unittest.main()
