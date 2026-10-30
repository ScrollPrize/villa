"""How remote arrays are requested: their request URLs and credentials.

``auth`` (of :meth:`ZarrReader.open` and :meth:`ZarrGroup.open`) is one of:

* ``"auto"`` (the default): ``s3://`` URLs of buckets without a known public
  https form are signed with AWS credentials from the environment, as with
  ``"aws"``, when there are any. Everything else is read anonymously.
* ``None``: anonymous.
* ``"aws"``: AWS credentials from the environment:
  ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``/``AWS_SESSION_TOKEN``, then
  the ``AWS_PROFILE`` (or default) profile of the shared credentials and config
  files. They are read once per open: long runs on temporary keys need reopening.
  SSO, assumed roles and instance metadata are not supported.
* an :class:`S3Credentials`: exactly those keys.
* ``(username, password)``: HTTP basic auth, for ``http(s)://`` URLs.

Signed S3 requests go to ``https://<bucket>.s3.<region>.amazonaws.com/<key>``
(path style for bucket names with dots, and for ``endpoint_url``). The region
is the credentials' ``region`` if set, else the bucket's own region (asked
from S3 once per bucket), else ``AWS_REGION``/``AWS_DEFAULT_REGION``.

401 and 403 responses raise :class:`PermissionError`. Note that S3 answers
403, not 404, for a missing object when the credentials may not list the
bucket (``s3:ListBucket``), so reading sparse arrays from private buckets
needs that permission: absent chunks are only read as ``fill_value`` on 404.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .. import _reader


def is_remote(location: str) -> bool:
    return _reader.is_remote(location)


def http_url(url: str) -> str:
    """https URL for an ``s3://`` URL, otherwise unchanged."""
    return _reader.http_url(url)


@dataclass(frozen=True)
class S3Credentials:
    """AWS keys for signed S3 requests.

    ``region`` defaults to the bucket's region. ``endpoint_url`` points
    requests at an S3-compatible server (MinIO, Ceph, ...) with path-style URLs.
    """

    access_key_id: str
    secret_access_key: str = field(repr=False)
    session_token: str | None = field(default=None, repr=False)
    region: str | None = None
    endpoint_url: str | None = None


def resolve_access(location: str, auth: Any = "auto", *, offline: bool = False):
    """The native ``Access`` for a location and an ``auth`` argument (see the module docstring)."""
    native = _reader
    if isinstance(auth, native.Access):
        return auth
    if not is_remote(location) or auth is None or auth == "anonymous":
        return native.anonymous_access()
    s3 = location.startswith("s3://")
    if isinstance(auth, tuple):
        if len(auth) != 2:
            raise ValueError("HTTP basic auth is a (username, password) tuple")
        if s3:
            raise ValueError("HTTP basic auth needs an http(s):// URL, not s3://")
        return native.basic_access(str(auth[0]), str(auth[1]))
    if isinstance(auth, S3Credentials):
        return native.s3_access(location, auth.access_key_id, auth.secret_access_key, auth.session_token or "",
                                auth.region or "", auth.endpoint_url or "", offline)
    if auth in ("auto", "aws"):
        return native.resolve_access(location, auth, offline)
    raise TypeError("auth must be 'auto', 'aws', None, an S3Credentials or a (username, password) tuple")


def load_basic_auth(path: str | os.PathLike[str]) -> tuple[str, str]:
    """``(username, password)`` from a JSON file: ``{"username": ..., "password": ...}``.

    The same file vesuvius reads for ``volume_auth_json``: the fields may sit
    under an ``"auth"`` object, and ``"user"`` is accepted for ``"username"``.
    """
    return tuple(_reader.load_basic_auth(os.fspath(path)))


__all__ = ["S3Credentials", "http_url", "is_remote", "load_basic_auth", "resolve_access"]
