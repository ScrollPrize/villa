"""Determinism settings, checkpoint I/O, run logging and a one-plan lookahead loader shared by the trainers."""
import copy
from contextlib import contextmanager
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before CUDA initializes: deterministic cuBLAS

import torch  # noqa: E402


def configure_torch(seed=27):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def cpu_copy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_copy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [cpu_copy(v) for v in x]
    if isinstance(x, tuple):
        return tuple(cpu_copy(v) for v in x)
    return x


def rng_state(rng):
    return copy.deepcopy(rng.bit_generator.state)


def set_rng_state(rng, state):
    rng.bit_generator.state = copy.deepcopy(state)


@contextmanager
def atomic_file(path):
    """Unique temporary file beside the destination; publish only a complete, fsynced file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save(path, state):
    """Atomic torch.save."""
    with atomic_file(path) as f:
        torch.save(state, f)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def write_json(path, value):
    with atomic_file(path) as f:
        f.write((json.dumps(value, indent=1, default=str) + '\n').encode())


def create_file(path, data):
    """Publish `data` at `path` only if no file is there: a complete, fsynced temporary file hard-linked into place
    (atomic and exclusive, also between processes). Returns False if `path` already existed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            return False
        return True
    finally:
        os.unlink(name)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def check_inputs(state, inputs, path):
    """Refuse to resume checkpoint `path` unless the inputs it recorded (CT, labels, ...) equal `inputs`."""
    recorded = state.get('inputs')
    if recorded is None:
        raise ValueError(f'{path} records no input identities (written before they were recorded), so its inputs '
                         'cannot be verified; restart the stage instead of resuming it')
    changed = {k: dict(checkpoint=recorded.get(k), now=inputs.get(k))
               for k in sorted(set(recorded) | set(inputs)) if recorded.get(k) != inputs.get(k)}
    if changed:
        raise ValueError(f'{path} was trained with different inputs; resume with the same ones: {changed}')


def output_directory(path, resume=False):
    """A fresh run must not append to or overwrite an earlier run's files."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not resume and any(path.iterdir()):
        raise FileExistsError(f'{path} is not empty; use a fresh --out or --resume')
    return path


def truncate_jsonl(path, step):
    """Drop rows after `step` (and a torn last line) from a curve before a resumed run appends to it."""
    path = Path(path)
    if path.exists():
        rows = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row['step'] <= step:
                rows.append(line + '\n')
        with atomic_file(path) as f:
            f.write(''.join(rows).encode())


def append_jsonl(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row) + '\n')


class Lookahead:
    """Loads the next plan's CPU packet in a worker thread while the current update runs. The worker only reads
    data; every random draw happens on the main thread, so the plan stream does not depend on timing."""
    def __init__(self, load):
        self.load = load
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def submit(self, *args):
        assert self.pending is None
        self.pending = self.pool.submit(self.load, *args)

    def take(self):
        start = time.time()
        packet = self.pending.result()
        self.pending = None
        return packet, time.time() - start

    def close(self):
        self.pool.shutdown(wait=True)
