import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stage_app_copies(stage):
    """Files and directories a Dockerfile stage COPYs from the build context into /app."""
    names, inside = [], False
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        s = line.strip()
        if s.upper().startswith("FROM "):
            inside = s.lower().endswith(f" as {stage}")
            continue
        if inside and s.upper().startswith("COPY ") and "--from" not in s and s.split()[-1].startswith("/app"):
            names.extend(p.rstrip("/") for p in s.split()[1:-1])
    return names


def _imported_top_level(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


class GpuImageModulesTests(unittest.TestCase):
    def test_gpu_image_copies_every_local_module_it_imports(self):
        copied = _stage_app_copies("gpu")
        self.assertIn("entrypoint.py", copied)
        self.assertIn("inference.py", copied)
        local = {p.stem for p in ROOT.glob("*.py")}
        wanted = set()
        for name in copied:
            path = ROOT / name
            files = [path] if path.suffix == ".py" else sorted(path.rglob("*.py")) if path.is_dir() else []
            for py in files:
                wanted |= _imported_top_level(py)
        missing = sorted((wanted & local) - {Path(n).stem for n in copied})
        self.assertEqual(missing, [], "the gpu image imports local modules it does not COPY into /app")


if __name__ == "__main__":
    unittest.main()
