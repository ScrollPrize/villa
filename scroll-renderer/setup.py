# Vesuvius Challenge Team 2024
# integrated from ThaumatoAnakalyptor

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='scroll_renderer',
    version='0.0.2b',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'Pillow',
        'opencv-python',
        'scipy',
        'tifffile',
        'dask',
        'dask-image',
        'zarr',
        'einops',
        'torch==2.5.0',
        'pytorch-lightning',
        'open3d',
        'libigl',
        'vesuvius',
        'psutil'
    ],
    entry_points={
        'console_scripts': [
            'slim_uv = scroll_renderer.slim_uv:main',
            'mesh_to_surface = scroll_renderer.mesh_to_surface:main',
            'large_mesh_to_surface = scroll_renderer.large_mesh_to_surface:main',
            'finalize_mesh = scroll_renderer.finalize_mesh:main',
        ],
    },
    author='Vesuvius Challenge Team',
    author_email='team@scrollprize.org',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ScrollPrize/villa',
    description='A package for flattening and rendering 3D meshes of segments of the Herculaneum Papyri.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering',
    ],
)
