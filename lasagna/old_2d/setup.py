from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "monotone_norm",
        ["monotone_norm.cpp"],
    ),
]

setup(
    name="monotone_norm",
    version="0.0.1",
    description="Monotone normalized distance transform (pybind11 extension)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)