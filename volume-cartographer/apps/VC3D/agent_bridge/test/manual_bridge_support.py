"""Shared fixtures and reporting for the developer-run bridge checks."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from vc3d_process import VC3DProcess


REPO_ROOT = Path(__file__).resolve().parents[4]
assert (REPO_ROOT / "apps" / "VC3D").is_dir(), f"unexpected REPO_ROOT={REPO_ROOT}"

DEFAULT_VC3D_BIN = REPO_ROOT / "build-macos" / "bin" / "VC3D"

# LOCAL_VOLPKG_JSON / S3_VOLPKG_JSON are developer-local fixtures, not
# committed (no secrets, just filesystem paths -- but at least one is a
# personal absolute path as data, so it can't be checked in as-is). They
# default to sitting one directory above the repo root; override via env var
# if yours lives elsewhere. Recreate LOCAL_VOLPKG_JSON with:
#   {
#     "name": "s1_2um_ds2",
#     "volumes": ["volume-cartographer/test-data/s1_ds2.volpkg/volumes"],
#     "segments": ["volume-cartographer/test-data/s1_ds2.volpkg/traces"],
#     "output_segments": "volume-cartographer/test-data/s1_ds2.volpkg/traces",
#     "normal_grids": ["volume-cartographer/test-data/s1_ds2.volpkg/normalgrids_2um_ds2"],
#     "lasagna_datasets": [],
#     "version": 1
#   }
# (paths are relative to the monorepo root, i.e. REPO_ROOT.parent). S3_VOLPKG_JSON
# follows the same schema but points "volumes" at an s3:// URI and "segments" at
# test-data/PHercParis4_neural_tracing.volpkg/segments/... dirs.
LOCAL_VOLPKG_JSON = Path(
    os.environ.get("VC3D_TEST_LOCAL_VOLPKG", str(REPO_ROOT.parent / "test.volpkg.json"))
)
S3_VOLPKG_JSON = Path(
    os.environ.get(
        "VC3D_TEST_S3_VOLPKG", str(REPO_ROOT.parent / "PHercParis4_neural_tracing.volpkg.json")
    )
)

# Curated points from test-data/s1_ds2.volpkg/trace_params.json.
LOCAL_REAL_SEED = {"x": 4914.0, "y": 3539.0, "z": 9150.0}
LOCAL_RAW_VOLUME_ID = "s1_2.4um_ds2_raw"

# Re-verification uses a distinct point so runs cover a fresh neighborhood.
LOCAL_REAL_SEED_OFFSCREEN_REVERIFY = {"x": 4326.0, "y": 4921.0, "z": 16350.0}

# Real point derived from an on-disk segment bbox center for the S3 fixture
# (test-data/PHercParis4_neural_tracing.volpkg/segments/extensions/w00_flat_clean/meta.json).
S3_REAL_POINT = {"x": 17500.0, "y": 19000.0, "z": 60000.0}

# Open Data manifest URL -- must match vc3d::opendata::kDefaultManifestUrl
# (apps/VC3D/OpenDataManifest.hpp). Used both for the reachability probe and to
# reason about what catalog.list_samples/describe_sample should return.
OPEN_DATA_MANIFEST_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json"
)


def probe_manifest_reachable(timeout: float = 8.0) -> tuple[bool, str]:
    """Quick network probe of the Open Data manifest URL. Returns
    (reachable, detail). A HEAD-like GET with a tiny read is enough to confirm
    the endpoint answers; we do not parse the (large) body here."""
    try:
        req = urllib.request.Request(OPEN_DATA_MANIFEST_URL, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            resp.read(64)  # touch the stream; don't download the whole manifest
            return (200 <= int(code) < 300, f"HTTP {code}")
    except Exception as e:  # noqa: BLE001 - any failure means "not reachable here"
        return (False, f"{type(e).__name__}: {e}")


def log(msg: str) -> None:
    print(f"[driver] {msg}", file=sys.stderr, flush=True)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def ms(seconds: float) -> float:
    return seconds * 1000.0


class Recorder:
    """Accumulates named pass/fail steps for the final report."""

    def __init__(self):
        self.steps: list[dict] = []
        self.failed = False

    def step(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append({"name": name, "ok": ok, "detail": detail})
        status = "OK  " if ok else "FAIL"
        log(f"{status} {name}: {detail}")
        if not ok:
            self.failed = True

    def as_dict(self) -> dict:
        return {"passed": not self.failed, "steps": self.steps}


class VC3DDiedError(Exception):
    """Abort a manual domain suite after its VC3D process exits."""


def record_process_death(
    proc: VC3DProcess, rec: Recorder, step_name: str
) -> None:
    code = proc.exit_code()
    tail = "\n".join(proc.tail_log(40))
    rec.step(
        f"VC3D died during {step_name}",
        False,
        f"VC3D died during {step_name}: exit={code}\nlog tail:\n{tail}",
    )


def launch_vc3d(binary: Path, offscreen: bool, socket_name: str,
                 extra_env: dict | None = None) -> VC3DProcess:
    env = dict(extra_env or {})
    if offscreen:
        env["QT_QPA_PLATFORM"] = "offscreen"
    args = ["--agent-bridge-name", socket_name]
    log(f"launching {binary} {' '.join(args)} (offscreen={offscreen})")
    return VC3DProcess(str(binary), args, env_overrides=env)
