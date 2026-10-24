"""Portable launcher for the shared native AFV constraint exporter."""
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from lasagna.inference_provenance import atomic_write, sha256_file


def resolve_native_binary(name, settings):
    """Resolve installed or caller-specified binaries without build-tree guesses."""
    default = str(Path(settings['binary_dir']) / name) if settings.get('binary_dir') else name
    binary = str(settings.get('native_binaries', {}).get(name, default))
    if name == 'vc_fiber_constraints':
        binary = str(settings.get('native_binary', binary))
    resolved = shutil.which(binary)
    if resolved is None and Path(binary).is_file():
        resolved = str(Path(binary).resolve())
    if resolved is None:
        raise FileNotFoundError(f'Build {name} and put it on PATH, or set native_binaries/binary_dir')
    return resolved


def afv_object_store(afv, settings):
    """The object store holding ``afv`` and its collection UUID. vc_fiber_constraints reads fibers only from an
    object store; each AFV is imported once with vc_object_store's import_afv.py (``afv_importer``: its path, or
    import_afv.py on PATH, run by this interpreter, which must import vc_object_store) into a store named by the
    AFV's content hash, next to the AFV or in ``afv_store_dir``. The AFV itself is never modified."""
    afv = Path(afv).resolve()
    store = Path(settings.get('afv_store_dir', afv.parent)) / f'{afv.stem}.{sha256_file(afv)[:16]}.vcobj'
    if not store.exists():
        importer = settings.get('afv_importer') or shutil.which('import_afv.py')
        if not importer or not Path(importer).is_file():
            raise FileNotFoundError('Set afv_importer to vc_object_store/scripts/import_afv.py, or put it on PATH')
        store.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(importer), str(afv), str(store), '--unbound'], check=True)
    with sqlite3.connect(afv.as_uri() + '?mode=ro', uri=True) as db:
        collection = json.loads(db.execute("SELECT value FROM metadata WHERE key='uuid'").fetchone()[0])
    return store, collection


def native_config_with_store(config_path, settings):
    """``config_path``, or for a config naming an ``afv`` a sibling config that also names its object store (the
    ``afv`` entry stays, as the run's recorded provenance)."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if 'afv' not in config or 'object_store' in config:
        return config_path
    store, collection = afv_object_store(config['afv'], settings)
    resolved = config_path.with_name(config_path.stem + '.store.json')
    atomic_write(resolved, dict(config, object_store=str(store), collection=collection), durable=False)
    return resolved


def run_native_constraints(config_path, settings):
    resolved = resolve_native_binary('vc_fiber_constraints', settings)
    if settings.get('native_regression_binary'):
        check = resolve_native_binary(str(settings['native_regression_binary']), settings)
        subprocess.run([check], check=True)
    subprocess.run([resolved, str(native_config_with_store(config_path, settings))], check=True)
