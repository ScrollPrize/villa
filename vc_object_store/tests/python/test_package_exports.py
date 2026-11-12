"""Exercise the installed package, not only in-tree aliases."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


def test_installed_server_provisions_its_dependencies():
    def run(*args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr

    python_path = os.environ.get('VC_OBJECT_STORE_BUILD_PYTHON_PATH')
    if not python_path or not shutil.which('cmake'):
        pytest.skip('needs the native CMake build')
    build = Path(python_path).resolve().parent
    if not (build / 'src' / 'libvc_object_server.a').exists():
        pytest.skip('needs the server target')
    with tempfile.TemporaryDirectory(prefix='vcobj-package-') as work:
        root = Path(work)
        prefix = root / 'install'
        run('cmake', '--install', str(build), '--prefix', str(prefix))
        source = root / 'consumer'
        source.mkdir()
        (source / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.20)
project(installed_consumer LANGUAGES C CXX)
find_package(vc_object_store CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE vc_object_store::vc_object_server)
''')
        (source / 'main.cpp').write_text('int main() { return 0; }\n')
        run('cmake', '-S', str(source), '-B', str(root / 'build'), '-DCMAKE_PREFIX_PATH=' + str(prefix))
        run('cmake', '--build', str(root / 'build'), '--parallel', '2')
