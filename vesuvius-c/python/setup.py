from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os

class CustomBuildExt(build_ext):
    def run(self):
        # Build libvesuvius.so
        base_dir = os.path.dirname(__file__)
        lib_path = os.path.join(base_dir, 'libvesuvius.so')
        cmd = [
            "gcc", "-shared", "-fPIC", "-O3",
            "-I..",
            "-o", lib_path,
            os.path.join(base_dir, "vesuvius_c_impl.c"),
            "-lcurl", "-lblosc2", "-ljson-c", "-lm"
        ]
        print(f"Building shared library: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        super().run()

setup(
    name="vesuvius-c-python",
    version="0.1.0",
    description="Python bindings for the Vesuvius-C library",
    packages=find_packages(),
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=[
        "numpy",
    ],
    include_package_data=True,
    package_data={
        '': ['libvesuvius.so'],
    },
)
