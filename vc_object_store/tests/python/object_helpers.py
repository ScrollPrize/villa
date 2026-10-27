import os
from pathlib import Path
import subprocess

import pytest


def raw(db, revision, name):
    """The stored bytes of one array of a geometry revision."""
    return db.read_geometry_arrays([revision], name).values.tobytes()


def export(db, uuid, output, revision=None, valid_only=False):
    """Export an object with the vc_object_export tool, as users do."""
    build = os.environ.get('VC_OBJECT_STORE_BUILD_PYTHON_PATH')
    tool = Path(build).resolve().parent / 'bin' / 'vc_object_export' if build else None
    if not tool or not tool.exists():
        pytest.skip('needs the built vc_object_export')
    command = [str(tool), '--db', db.path, '--uuid', uuid, '--output', str(output)]
    if revision:
        command += ['--revision', revision]
    if valid_only:
        command.append('--valid-only')
    return subprocess.run(command, capture_output=True, text=True)
