import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import warnings

version = os.environ.get("VERSION", "0.1.10")

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        message = """
        ============================================================
        Thank you for installing vesuvius!

        To complete the setup, please run the following command:

            vesuvius.accept_terms --yes

        This will display the terms and conditions to be accepted.
        ============================================================
        """
        warnings.warn(message, UserWarning)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='vesuvius',
    version=version,
    py_modules=['vesuvius'],
    packages=find_packages(),
    url='https://github.com/ScrollPrize/villa',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'requests',
        'aiohttp',
        'fsspec',
        'tensorstore',
        'huggingface_hub',
        'dask',
        'zarr',
        'tqdm',
        'lxml',
        'nest_asyncio',
        'pynrrd',
        'pyyaml',
        'Pillow',
        'Torch',
        'nnUNetv2',
        'scipy',
        'batchgenerators',
        'batchgeneratorsv2',
        'dynamic_network_architectures'
    ],
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        'vesuvius': ['setup/configs/*.yaml'],
        'setup': ['configs/*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'vesuvius.accept_terms=setup.accept_terms:main',
            'vesuvius.predict=models.run.inference:main',
            'vesuvius.blend_logits=models.run.blending:main',
            'vesuvius.finalize_outputs=models.run.finalize_outputs:main',
            'vesuvius.inference_pipeline=models.run.vesuvius_pipeline:run_pipeline',
        ],
    },
    # No scripts needed as we're using entry_points
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering',
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)
