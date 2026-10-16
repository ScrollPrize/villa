"""Checkpoints on the Hugging Face Hub: ``--checkpoint ORG/NAME`` (also ``hf://ORG/NAME``, with ``@REVISION`` for a
branch, tag or commit) instead of a local file, downloaded into the Hugging Face cache (``$HF_HOME``). Each tracer
picks its file(s) from the downloaded folder (``fibers/tracing/runtime.py``, ``sheets/tracing/inference.py``)."""
from __future__ import annotations

import os
import re
from pathlib import Path


def hf_repo(checkpoint):
    """(repo id, revision or None) when ``checkpoint`` names a Hugging Face repo rather than a local file, else None."""
    text = str(checkpoint)
    if text.startswith('hf://'):
        text = text[len('hf://'):]
    elif Path(text).expanduser().exists():
        return None
    match = re.fullmatch(r'([\w.-]+/[\w.-]+)(?:@(\S+))?', text)
    if not match or match[1].endswith('.pt'):
        return None
    return match[1], match[2]


def hub_folder(repo, patterns=('*.pt',)) -> Path:
    """The local snapshot folder of ``repo`` (``hf_repo``'s (repo id, revision)) with its files matching ``patterns``
    (default: its ``.pt`` files)."""
    from huggingface_hub import snapshot_download
    repo_id, revision = repo
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=list(patterns)))


def checkpoint_source(checkpoint, path):
    """What a run records as its checkpoint: hf://REPO@COMMIT/FILE for a Hugging Face repo, else the absolute path."""
    repo = hf_repo(checkpoint)
    if repo is None:
        return os.path.abspath(path)
    return f'hf://{repo[0]}@{Path(path).parent.name}/{Path(path).name}'  # the snapshot folder is the commit
