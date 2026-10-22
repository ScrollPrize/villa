"""Portable launcher for the shared native AFV constraint exporter."""
from pathlib import Path
import shutil
import subprocess


def run_native_constraints(config_path, settings):
    binary = str(settings.get('native_binary', 'vc_fiber_constraints'))
    resolved = shutil.which(binary)
    if resolved is None and Path(binary).is_file():
        resolved = str(Path(binary).resolve())
    if resolved is None:
        raise FileNotFoundError('Build vc_fiber_constraints and put it on PATH, or set native_binary')
    subprocess.run([resolved, str(Path(config_path).resolve())], check=True)
