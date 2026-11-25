"""The packaged inventory of instance-annotated NRRD cubes.

Scroll volumes and segments are discovered through vc_reader's Open Data catalog.
"""

from importlib.resources import files

import yaml


def list_cubes() -> dict:
    """Return scroll → energy → resolution → cube origin → base URL."""
    resource = files('vesuvius').joinpath('install', 'configs', 'cubes.yaml')
    return yaml.safe_load(resource.read_text(encoding='utf-8'))
