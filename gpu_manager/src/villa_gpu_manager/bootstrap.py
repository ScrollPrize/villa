"""Connect to a local manager, starting one safely when it is absent."""

import errno
import grp
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys
import time

import psutil

from .client import Client
from .paths import default_socket_path


def visible_gpu_uuids():
    """Resolve CUDA visibility without creating a CUDA context."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True, timeout=10)
    rows = [tuple(part.strip() for part in line.split(",")) for line in output.splitlines() if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [uuid for _, uuid in rows]
    if visible.strip() in ("", "-1"):
        return []
    result = []
    for token in visible.split(","):
        token = token.strip()
        if not token or token == "-1":
            raise ValueError("CUDA_VISIBLE_DEVICES contains an empty or disabled entry in a GPU list")
        matches = [uuid for index, uuid in rows if token == index or uuid.startswith(token)]
        if len(matches) != 1 or matches[0] in result:
            raise ValueError(f"CUDA_VISIBLE_DEVICES contains unsupported/ambiguous GPU {token!r}")
        result.append(matches[0])
    if any(token.strip().isdigit() for token in visible.split(",")):
        print(f"gpu manager: visibility {visible!r} resolved as nvidia-smi physical indices to {result}; "
              "GPU workers use UUIDs (use a UUID mask to avoid ordinal ambiguity)", flush=True)
    return result


def _inventory(path):
    with Client(path, timeout=1) as client:
        return client.status()["inventory"]


def compatible_gpu_uuids(names):
    """Visible devices whose model was actually covered by calibration."""
    visible = set(visible_gpu_uuids())
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True, timeout=10)
    rows = [line.split(",", 1) for line in output.splitlines() if line.strip()]
    return [uuid.strip() for uuid, name in rows if uuid.strip() in visible and name.strip() in names]


def _shared_uids(path):
    parent = Path(path).parent
    if not parent.exists():
        return None
    info = parent.stat()
    if not stat.S_IMODE(info.st_mode) & 0o070:
        return None
    # Server validates owner, sticky bit and absence of world permissions.
    members = set(grp.getgrgid(info.st_gid).gr_mem)
    return sorted({os.getuid()} | {p.pw_uid for p in pwd.getpwall()
                                  if p.pw_gid == info.st_gid or p.pw_name in members})


def ensure_manager(path=None, *, timeout=20.0):
    """Return (socket, inventory); never replace an incompatible/unauthorized service.

    Competing starters are serialized by Server's existing flock and journal.
    The detached manager exits after its last client/task leaves. No CUDA is used.
    """
    path = str(Path(path or default_socket_path()).expanduser())
    try:
        return path, _inventory(path)
    except OSError as exc:
        if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED):
            raise
    log_dir = Path.home() / ".local/state/villa-gpu-manager"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    import hashlib
    log = log_dir / (hashlib.sha256(path.encode()).hexdigest()[:16] + ".log")
    command = [sys.executable, "-m", "villa_gpu_manager.bootstrap", "--serve", path]
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("ab") as output:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, env=env, start_new_session=True)
    print(f"gpu manager: starting socket={path} log={log}", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return path, _inventory(path)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED):
                raise
        # A concurrent winner may still be binding after this starter lost the lock.
        child.poll()
        time.sleep(0.1)
    raise RuntimeError(f"manager did not start at {path}; inspect {log} (no fallback or journal reset)")


def main():
    import argparse
    from .cli import main as manager_main
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", required=True)
    args = parser.parse_args()
    # The server describes the host, not only the first client's visibility.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    devices = visible_gpu_uuids()
    if not devices:
        raise RuntimeError("no NVIDIA GPUs found")
    cores = psutil.cpu_count(logical=True)
    if not cores:
        raise RuntimeError("cannot determine logical CPU count")
    command = ["--socket", args.serve, "serve", "--gpus", *devices, "--cpu-cores", str(cores)]
    uids = _shared_uids(args.serve)
    if uids is not None:
        command.append("--shared-group")
        for uid in uids:
            command.extend(["--allow-uid", str(uid)])
    return manager_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
