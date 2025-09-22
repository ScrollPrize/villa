"""Provide the package version for setuptools dynamic metadata."""

from __future__ import annotations

import os

VERSION: str = os.environ.get("VERSION", "0.1.10")
"""Current package version, optionally overridden via the VERSION env var."""

__all__ = ["VERSION"]
