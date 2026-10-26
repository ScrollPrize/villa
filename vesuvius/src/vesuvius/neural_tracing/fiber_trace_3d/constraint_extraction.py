"""Portable launcher for the shared native object-store fiber constraint exporter."""
from pathlib import Path
import shutil
import subprocess


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


def run_native_constraints(config_path, settings):
    resolved = resolve_native_binary('vc_fiber_constraints', settings)
    if settings.get('native_regression_binary'):
        check = resolve_native_binary(str(settings['native_regression_binary']), settings)
        subprocess.run([check], check=True)
    subprocess.run([resolved, str(Path(config_path).resolve())], check=True)
