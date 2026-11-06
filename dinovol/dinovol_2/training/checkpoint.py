"""Versioned checkpoint readers shared by training and inference."""
from __future__ import annotations

from typing import Any, Mapping

FORMAT_VERSION = 2


def method_state(checkpoint: Mapping[str, Any], *, expected_method: str | None = None):
    version = checkpoint.get('format_version', 1)
    if version == FORMAT_VERSION:
        method = checkpoint['method']
        name, state = method['name'], method['state']
    elif version == 1:
        if 'student' not in checkpoint and 'teacher' not in checkpoint:
            raise ValueError('Legacy training checkpoint has no DINO model branches')
        name, state = 'dino', checkpoint
    else:
        raise ValueError(f'Unsupported checkpoint format version: {version}')
    if expected_method is not None and name != expected_method:
        raise ValueError(f'Checkpoint method {name!r} does not match requested method {expected_method!r}')
    return state


def backbone_state(checkpoint: Mapping[str, Any], *, preferred_branch='teacher'):
    """Return backbone-only weights and their source, including legacy DINO files."""
    state = method_state(checkpoint)
    order = list(dict.fromkeys((preferred_branch, 'teacher', 'student')))
    for branch_name in order:
        branch = state.get(branch_name)
        if isinstance(branch, Mapping):
            backbone = {key.removeprefix('backbone.'): value for key, value in branch.items()
                        if key.startswith('backbone.')}
            if backbone:
                return backbone, branch_name
    exported = checkpoint.get('evaluation_backbone')
    if exported is not None:
        return exported, 'evaluation_backbone'
    raise ValueError('Checkpoint has no exported evaluation backbone or DINO backbone weights')
