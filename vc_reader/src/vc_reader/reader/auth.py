"""How remote arrays are requested: their request URLs and credentials.

``auth`` (of :meth:`ZarrReader.open` and :meth:`ZarrGroup.open`) is one of:

* ``"auto"`` (the default): ``s3://`` URLs of buckets without a known public
  https form are signed with AWS credentials from the environment, as with
  ``"aws"``, when there are any. Everything else is read anonymously.
* ``None``: anonymous.
* ``"aws"``: AWS credentials from the environment, the way boto finds them:
  ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``/``AWS_SESSION_TOKEN``, then
  the ``AWS_PROFILE`` (or default) profile of the shared credentials and config
  files. With botocore installed, also SSO, assumed roles and instance
  metadata. Temporary credentials are refreshed as they near expiry.
* an :class:`S3Credentials`: exactly those keys (never refreshed).
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

import configparser
import functools
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Seconds between checks of refreshable ("aws") credentials.
REFRESH_INTERVAL_S = 60.0

ANONYMOUS = ("", "", "", "", "")


def is_remote(location: str) -> bool:
    return location.startswith(("http://", "https://", "s3://"))


def http_url(url: str) -> str:
    """https URL for an ``s3://`` URL (public buckets only), otherwise unchanged."""
    if not url.startswith("s3://"):
        return url
    from ..artifacts import to_https

    https = to_https(url)
    if https is None:
        bucket, _, key = url[len("s3://"):].partition("/")
        https = f"https://{bucket}.s3.amazonaws.com/{key}"
    return https


def split_s3(url: str) -> tuple[str, str]:
    """``s3://bucket/key`` -> ``(bucket, key)``."""
    bucket, _, key = url[len("s3://"):].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {url!r}")
    return bucket, key


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

    @classmethod
    def from_environment(cls, profile: str | None = None) -> S3Credentials | None:
        """A snapshot of the credentials ``auth="aws"`` would use (``profile`` overrides ``AWS_PROFILE``), or None."""
        source = _AwsSource(profile)
        return source.current()


def _config_sections(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read(path, encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError):
        pass
    return parser


def _aws_file(env: str, default: str) -> Path:
    return Path(os.environ.get(env) or Path.home() / ".aws" / default).expanduser()


def _profile_setting(profile: str, key: str) -> str | None:
    """``key`` of ``profile`` in the shared config file (``[profile x]`` or ``[default]``)."""
    config = _config_sections(_aws_file("AWS_CONFIG_FILE", "config"))
    section = "default" if profile == "default" else f"profile {profile}"
    value = config.get(section, key, fallback=None)
    return value.strip() if value else None


def _env_region() -> str | None:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None


class _AwsSource:
    """Credentials from the AWS environment; with botocore, refreshed the way botocore refreshes them."""

    def __init__(self, profile: str | None = None):
        self.profile = profile
        self._botocore: Any = None  # botocore Credentials, which refresh themselves
        self._static: S3Credentials | None = None
        self._region: str | None = None
        self._resolved = False

    def _profile(self) -> str:
        return self.profile or os.environ.get("AWS_PROFILE") or os.environ.get("AWS_DEFAULT_PROFILE") or "default"

    def _resolve(self) -> None:
        self._resolved = True
        try:
            import botocore.session
        except ImportError:
            self._static = self._from_files()
            self._region = _profile_setting(self._profile(), "region")
            return
        session = botocore.session.Session(profile=self.profile)
        self._botocore = session.get_credentials()
        self._region = session.get_config_variable("region")

    def _from_files(self) -> S3Credentials | None:
        """The environment variables, then the shared credentials and config files (stdlib only)."""
        env_key, env_secret = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
        if self.profile is None and env_key and env_secret:
            return S3Credentials(env_key, env_secret, os.environ.get("AWS_SESSION_TOKEN") or None)
        profile = self._profile()
        creds = _config_sections(_aws_file("AWS_SHARED_CREDENTIALS_FILE", "credentials"))
        if creds.has_section(profile):
            key = creds.get(profile, "aws_access_key_id", fallback=None)
            secret = creds.get(profile, "aws_secret_access_key", fallback=None)
            if key and secret:
                token = creds.get(profile, "aws_session_token", fallback=None)
                return S3Credentials(key.strip(), secret.strip(), token.strip() if token else None)
        key = _profile_setting(profile, "aws_access_key_id")
        secret = _profile_setting(profile, "aws_secret_access_key")
        if key and secret:
            return S3Credentials(key, secret, _profile_setting(profile, "aws_session_token"))
        return None

    def current(self) -> S3Credentials | None:
        if not self._resolved:
            self._resolve()
        if self._botocore is None:
            return self._static
        frozen = self._botocore.get_frozen_credentials()  # refreshes if near expiry
        if not frozen.access_key or not frozen.secret_key:
            return None
        return S3Credentials(frozen.access_key, frozen.secret_key, frozen.token or None)

    def region(self) -> str | None:
        if not self._resolved:
            self._resolve()
        return self._region

    def __getstate__(self):
        return {"profile": self.profile}  # resolved again in the new process

    def __setstate__(self, state):
        self.__init__(state["profile"])


@functools.lru_cache(maxsize=256)
def bucket_region(bucket: str) -> str | None:
    """The bucket's region, from S3's ``x-amz-bucket-region`` header (an anonymous HEAD); None if unknown."""
    req = urllib.request.Request(f"https://{bucket}.s3.amazonaws.com/", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.headers.get("x-amz-bucket-region")
    except urllib.error.HTTPError as e:  # 403 (private) and 301 (other region) carry it too
        return e.headers.get("x-amz-bucket-region")
    except (urllib.error.URLError, OSError):
        return None


class Access:
    """How requests for locations under one root are made. Picklable.

    ``kind`` is ``"anonymous"``, ``"basic"`` or ``"s3"``. Signed (``"s3"``)
    access holds either fixed keys or an AWS source refreshed as it expires.
    """

    def __init__(self, kind: str, *, basic: tuple[str, str] | None = None, keys: S3Credentials | None = None,
                 source: _AwsSource | None = None, region: str | None = None, endpoint_url: str | None = None):
        self.kind = kind
        self.basic = basic
        self.keys = keys
        self.source = source
        self.region = region
        self.endpoint_url = endpoint_url.rstrip("/") if endpoint_url else None
        self._lock = threading.Lock()
        self._auth_tuple: tuple[str, str, str, str, str] | None = None
        self._next_check = 0.0

    def __repr__(self) -> str:
        if self.kind == "s3":
            how = "fixed keys" if self.keys is not None else "AWS environment"
            return f"Access(s3, {how}, region={self.region!r})"
        return f"Access({self.kind})"

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_lock"]
        if self.source is not None:  # refreshable: resolved again in the new process
            state["_auth_tuple"], state["_next_check"] = None, 0.0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # -- URLs ---------------------------------------------------------------------------

    def url(self, location: str) -> str:
        """The URL to GET for a remote location."""
        if self.kind != "s3":
            return http_url(location)
        bucket, key = split_s3(location)
        return self._bucket_root(bucket) + urllib.parse.quote(key, safe="/-_.~")

    def list_url(self, location: str, continuation: str | None = None) -> str:
        """ListObjectsV2 of the immediate children of an ``s3://`` location."""
        bucket, key = split_s3(location)
        prefix = key.rstrip("/") + "/" if key.strip("/") else ""
        query = {"list-type": "2", "delimiter": "/", "prefix": prefix}
        if continuation:
            query["continuation-token"] = continuation
        root = self._bucket_root(bucket) if self.kind == "s3" else f"https://{bucket}.s3.amazonaws.com/"
        return root + "?" + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)

    def _bucket_root(self, bucket: str) -> str:
        if self.endpoint_url:
            return f"{self.endpoint_url}/{bucket}/"
        if "." in bucket:  # virtual-host names with dots fail TLS verification
            return f"https://s3.{self.region}.amazonaws.com/{bucket}/"
        return f"https://{bucket}.s3.{self.region}.amazonaws.com/"

    # -- credentials ----------------------------------------------------------------------

    def auth_tuple(self) -> tuple[str, str, str, str, str]:
        """The reader authentication tuple, with refreshable credentials checked at most every REFRESH_INTERVAL_S."""
        if self.kind == "anonymous":
            return ANONYMOUS
        if self.kind == "basic":
            assert self.basic is not None
            return ("basic", self.basic[0], self.basic[1], "", "")
        if self._auth_tuple is not None and (self.source is None or time.monotonic() < self._next_check):
            return self._auth_tuple
        with self._lock:
            if self._auth_tuple is None or (self.source is not None and time.monotonic() >= self._next_check):
                keys = self.keys if self.source is None else self.source.current()
                if keys is None:
                    raise PermissionError("no AWS credentials found (environment, AWS_PROFILE or ~/.aws)")
                self._auth_tuple = ("s3", keys.access_key_id, keys.secret_access_key, keys.session_token or "",
                                self.region or "us-east-1")
                self._next_check = time.monotonic() + REFRESH_INTERVAL_S
            return self._auth_tuple


_ANON = Access("anonymous")


def resolve_access(location: str, auth: Any = "auto") -> Access:
    """The :class:`Access` for a location and an ``auth`` argument (see the module docstring)."""
    if not is_remote(location) or auth is None or auth == "anonymous":
        return _ANON
    if isinstance(auth, Access):
        return auth
    s3 = location.startswith("s3://")
    if isinstance(auth, tuple):
        if len(auth) != 2:
            raise ValueError("HTTP basic auth is a (username, password) tuple")
        if s3:
            raise ValueError("HTTP basic auth needs an http(s):// URL, not s3://")
        return Access("basic", basic=(str(auth[0]), str(auth[1])))
    if auth == "auto":
        from ..artifacts import to_https

        if not s3 or to_https(location) is not None:
            return _ANON
        source = _AwsSource()
        if source.current() is None:  # no credentials: anonymous, as for a public bucket
            return _ANON
        return _s3_access(location, source=source)
    if auth == "aws":
        if not s3:
            raise ValueError(f"AWS credentials need an s3:// URL, not {location!r}")
        source = _AwsSource()
        if source.current() is None:
            raise PermissionError("auth='aws' but no AWS credentials were found (environment, AWS_PROFILE or ~/.aws)")
        return _s3_access(location, source=source)
    if isinstance(auth, S3Credentials):
        if not s3:
            raise ValueError(f"S3 credentials need an s3:// URL, not {location!r}")
        return _s3_access(location, keys=auth)
    raise TypeError("auth must be 'auto', 'aws', None, an S3Credentials or a (username, password) tuple")


def _s3_access(location: str, *, keys: S3Credentials | None = None, source: _AwsSource | None = None) -> Access:
    bucket, _ = split_s3(location)
    endpoint = (keys.endpoint_url if keys else None) or os.environ.get("AWS_ENDPOINT_URL_S3") \
        or os.environ.get("AWS_ENDPOINT_URL")
    region = keys.region if keys else None
    if region is None and not endpoint:
        region = bucket_region(bucket)
    if region is None:
        region = _env_region() or (source.region() if source else None) or "us-east-1"
    return Access("s3", keys=keys, source=source, region=region, endpoint_url=endpoint)


def load_basic_auth(path: str | os.PathLike[str]) -> tuple[str, str]:
    """``(username, password)`` from a JSON file: ``{"username": ..., "password": ...}``.

    The same file vesuvius reads for ``volume_auth_json``: the fields may sit
    under an ``"auth"`` object, and ``"user"`` is accepted for ``"username"``.
    """
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    root = data["auth"] if isinstance(data.get("auth"), dict) else data
    user, password = root.get("username", root.get("user")), root.get("password")
    if not user or password is None:
        raise ValueError(f"{path}: needs 'username' (or 'user') and 'password'")
    return str(user), str(password)


__all__ = ["Access", "S3Credentials", "http_url", "is_remote", "load_basic_auth", "resolve_access"]
