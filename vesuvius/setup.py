from setuptools import setup, find_packages
from setuptools.command.install import install
import warnings

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
    version='0.1.10',
    package_dir = {"": "src"},
    packages=find_packages(where="src"),
    url='https://github.com/ScrollPrize/villa',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'requests',
        'aiohttp',
        'fsspec',
        'tensorstore',
        'zarr',
        'tqdm',
        'lxml',
        'nest_asyncio',
        'pynrrd',
        'pyyaml',
        'Pillow'
    ],
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        '': ['src/vesuvius/configs/*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'vesuvius.accept_terms=vesuvius.setup.accept_terms:main',
        ],
    },
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
