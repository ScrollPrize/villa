"""JSON output with numpy values."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def write_json(path, value):
    """Write ``value`` (numpy values allowed) as indented JSON, atomically, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, default=jsonable, indent=1)+'\n')
    temporary.replace(path)


