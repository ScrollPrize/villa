"""Zarr 2/3 compatibility for array creation.

``pyproject.toml`` allows ``zarr>=2.18.7,<4``, and the package already branches
on ``_ZARR_V3`` elsewhere, but many modules still call ``Group.create_dataset``
and ``Group.require_dataset``. Zarr 3 removed both, so those code paths raise
``AttributeError`` on a zarr 3 install -- including the declared console script
``vesuvius.compute_st``.

A blind rename to ``create_array`` / ``require_array`` would break zarr 2 users,
who are still supported by the version pin. This module keeps one call shape
working on both.

What actually differs
---------------------
Tested against zarr 3.3.0; most keywords pass straight through, and only three
need translating:

===========================  ===========================================
zarr 2 keyword               zarr 3 equivalent
===========================  ===========================================
``write_empty_chunks=B``     ``config={"write_empty_chunks": B}``
``dimension_separator=S``    ``chunk_key_encoding=V2ChunkKeyEncoding(S)``
``chunks=None``              ``chunks="auto"`` (``None`` is rejected)
===========================  ===========================================

``shape``, ``dtype``, ``data``, ``chunks``, ``fill_value``, ``overwrite`` and
``compressor`` (including a numcodecs instance) are accepted unchanged.

The two creation paths disagree
-------------------------------
This is the part a naive port gets wrong. ``Group.create_array`` and top-level
``zarr.open`` accept *opposite* keywords for the same intent:

=========================  =======================  =========================
keyword                    ``Group.create_array``   top-level ``zarr.open``
=========================  =======================  =========================
``dimension_separator``    TypeError                accepted (nests keys)
``chunk_key_encoding``     accepted                 ValueError on zarr_format 2
``write_empty_chunks``     TypeError                accepted but deprecated
``config={...}``           accepted                 accepted
=========================  =======================  =========================

So one translation table cannot serve both. :func:`create_array` and
:func:`require_array` handle the group path; :func:`open_array` handles the
top-level path, which is what the ``NestedDirectoryStore`` call sites use.
``config`` is the only spelling both accept, so it is preferred for
``write_empty_chunks`` on each.

Why translate rather than drop
------------------------------
``write_empty_chunks=False`` is a storage-size decision -- silently dropping it
would make sparse label volumes materialise every empty chunk, which is a
regression that no test would notice. ``dimension_separator`` decides the
on-disk key layout, so dropping it changes where chunks are written and makes
previously-written data unreadable. Both are translated, not ignored.
"""

from __future__ import annotations

from typing import Any

import zarr

__all__ = ["ZARR_MAJOR", "create_array", "require_array", "local_store",
           "open_array", "nested_store_and_kwargs"]

ZARR_MAJOR = int(str(zarr.__version__).split(".")[0])


def _translate_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map zarr 2 creation keywords onto their zarr 3 equivalents."""
    out = dict(kwargs)

    # zarr 2 treated chunks=None as "auto"; zarr 3 rejects None outright.
    if "chunks" in out and out["chunks"] is None:
        out["chunks"] = "auto"

    if "write_empty_chunks" in out:
        value = out.pop("write_empty_chunks")
        config = dict(out.get("config") or {})
        config.setdefault("write_empty_chunks", value)
        out["config"] = config

    if "dimension_separator" in out:
        separator = out.pop("dimension_separator")
        if separator is not None and "chunk_key_encoding" not in out:
            from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

            out["chunk_key_encoding"] = V2ChunkKeyEncoding(separator=separator)

    return out


def create_array(group: Any, name: str, **kwargs: Any) -> Any:
    """``group.create_dataset(name, **kwargs)`` that works on zarr 2 and 3."""
    if ZARR_MAJOR < 3:
        return group.create_dataset(name, **kwargs)
    return group.create_array(name, **_translate_kwargs(kwargs))


def require_array(group: Any, name: str, **kwargs: Any) -> Any:
    """``group.require_dataset(name, **kwargs)`` that works on zarr 2 and 3.

    Zarr 3's ``require_array`` requires ``shape`` as a keyword, which every
    call site in this package already supplies.
    """
    if ZARR_MAJOR < 3:
        return group.require_dataset(name, **kwargs)
    return group.require_array(name, **_translate_kwargs(kwargs))


def _translate_open_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map zarr 2 creation keywords for the TOP-LEVEL ``zarr.open`` path.

    Deliberately different from :func:`_translate_kwargs`; see the module
    docstring for why the two paths cannot share one table. Here
    ``dimension_separator`` is kept (it is the accepted spelling) while
    ``write_empty_chunks`` still moves into ``config`` to avoid the deprecation
    warning zarr 3 emits for the bare keyword.
    """
    out = dict(kwargs)

    if "chunks" in out and out["chunks"] is None:
        out["chunks"] = "auto"

    if "write_empty_chunks" in out:
        value = out.pop("write_empty_chunks")
        config = dict(out.get("config") or {})
        config.setdefault("write_empty_chunks", value)
        out["config"] = config

    # dimension_separator is left alone: zarr 3 accepts it here and REJECTS
    # chunk_key_encoding for zarr_format 2.
    return out


def open_array(*, store: Any, mode: str = "w", **kwargs: Any) -> Any:
    """``zarr.open(store=..., mode=..., **creation_kwargs)`` for zarr 2 and 3."""
    if ZARR_MAJOR < 3:
        return zarr.open(store=store, mode=mode, **kwargs)
    return zarr.open(store=store, mode=mode, **_translate_open_kwargs(kwargs))


def local_store(path: str, *, nested: bool = False, **kwargs: Any) -> Any:
    """``zarr.DirectoryStore`` / ``zarr.NestedDirectoryStore`` replacement.

    Zarr 3 removed both in favour of ``zarr.storage.LocalStore``. The nested
    chunk layout is no longer a property of the store -- it comes from the
    array's key encoding -- so on zarr 3 ``nested=True`` returns a plain
    ``LocalStore`` and the caller must also pass ``dimension_separator="/"``
    to :func:`open_array`. :func:`nested_store_and_kwargs` does both.
    """
    if ZARR_MAJOR < 3:
        cls = zarr.NestedDirectoryStore if nested else zarr.DirectoryStore
        return cls(path, **kwargs)
    from zarr.storage import LocalStore

    return LocalStore(path, **kwargs)


def nested_store_and_kwargs(path: str) -> tuple[Any, dict[str, Any]]:
    """Store plus creation kwargs reproducing ``NestedDirectoryStore`` layout.

    On zarr 2 the store carries the nesting and no extra kwargs are needed. On
    zarr 3 the store is plain and the nesting must be requested per array, so
    forgetting the second half silently writes a flat layout that older readers
    will not find.
    """
    if ZARR_MAJOR < 3:
        return zarr.NestedDirectoryStore(path), {}
    from zarr.storage import LocalStore

    # zarr_format=2 is required, not cosmetic. A "/" dimension separator is a
    # zarr v2 concept; zarr 3 defaults new arrays to format 3 and then rejects
    # the keyword outright ("dimension_separator cannot be used for arrays with
    # zarr_format 3"). Callers writing nested pyramids also hand-write a
    # .zgroup with zarr_format 2, so a format-3 array would sit inside a
    # format-2 group -- inconsistent even if it were accepted.
    return LocalStore(path), {"dimension_separator": "/", "zarr_format": 2}
