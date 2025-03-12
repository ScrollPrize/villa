from setuptools import setup, find_packages

setup(
    name="tracer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "theseus-ai>=0.1.0",  # Ensure version with CholmodSparseSolver support
        "matplotlib",
        "numpy",
        "pillow",
    ],
    author="Volume Cartographer Team",
    author_email="info@volumecartographer.com",
    description="Theseus implementation of cost functions from Volume Cartographer",
    keywords="optimization, nonlinear, differentiable, theseus",
    url="https://github.com/volumecartographer/vc_tracer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)