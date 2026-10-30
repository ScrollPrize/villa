"""Data entries ("artifacts") and where to find them (``vcr::catalog``).

Every entity in metadata.json carries a ``data`` list. Each entry names an
artifact ``type``, one or more ``origins`` (alternative locations of the same
content), the ``parameters`` that identify it among artifacts of the same type,
and free-form ``properties`` / ``creation_info``.
"""
from ._catalog import AccessRoot, Artifact, ArtifactType, Origin, join_url, parse_artifacts, to_https, to_s3
