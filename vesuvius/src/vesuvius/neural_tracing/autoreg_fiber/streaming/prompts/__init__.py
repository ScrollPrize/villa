"""Prompt-construction helpers for the streaming tracer.

The streaming tracer takes a fiber-cache ``.npz`` (the same format that
``build_fiber_cache.py`` produces). When the user already has one on disk,
no helper is needed; when they only have a WebKnossos annotation URL,
:func:`from_wk_url` downloads, picks a tree, and writes an ``.npz``
ready for the tracer.
"""

from vesuvius.neural_tracing.autoreg_fiber.streaming.prompts.from_wk_url import (
    build_npz_from_wk_url,
)

__all__ = ["build_npz_from_wk_url"]
