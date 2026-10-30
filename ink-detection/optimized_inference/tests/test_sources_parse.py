import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourcesParseTests(unittest.TestCase):
    def test_runtime_modules_parse(self):
        # entrypoint.py imports boto3/cv2/torch, so parse rather than import to stay dependency-free.
        for path in sorted(ROOT.glob("*.py")):
            with self.subTest(module=path.name):
                ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
