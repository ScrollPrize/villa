"""Hash the actual shared diagnostic implementation, not a workspace wrapper."""
import hashlib
from pathlib import Path

def code_sha256():
    root = Path(__file__).resolve().parent
    names = ("__init__.py", "__main__.py", "geometry.py", "slab.py",
             "capture.py", "cli.py", "provenance.py")
    return {f"vc_sampling_audit/{name}": hashlib.sha256(
        (root / name).read_bytes()).hexdigest() for name in names}
