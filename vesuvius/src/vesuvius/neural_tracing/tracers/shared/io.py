"""Atomic file output and JSON with numpy values."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def atomic_write(path, data):
    """Write ``data`` (bytes) to ``path`` through a unique temporary file and an atomic rename, creating parent folders;
    concurrent readers see the old or the new file, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.partial-',delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def write_json(path, value):
    """Write ``value`` (numpy values allowed) as indented JSON, atomically, creating parent folders."""
    atomic_write(path, (json.dumps(value, default=jsonable, indent=1)+'\n').encode())


def atomic_torch_save(value, path):
    """``torch.save`` through a temporary sibling (``.partial.pt``) and an atomic rename, so a reader (a collector,
    an evaluation) never opens a partial checkpoint."""
    import torch
    path = Path(path)
    temporary = path.with_suffix('.partial.pt')
    torch.save(value, temporary)
    temporary.replace(path)
