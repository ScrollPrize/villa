import os
from pathlib import Path
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


def get_local_package_path(relative_path):
    """Convert relative path to absolute file:// URL"""
    current_file = Path(__file__).resolve()
    target_path = (current_file.parent / relative_path).resolve()
    return f"file://{target_path}"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vesuvius",
    version=version,
    py_modules=["vesuvius"],
    packages=find_packages(),
    url="https://github.com/ScrollPrize/villa",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "requests",
        "aiohttp",
        "fsspec",
        "huggingface_hub",
        "zarr>=2,<3",
        "tqdm",
        "lxml",
        "nest_asyncio",
        "pynrrd",
        "pyyaml",
        "Pillow",
        "Torch",
        "scipy",
        "batchgenerators",
        f"batchgeneratorsv2 @ {get_local_package_path("../segmentation/models/batchgeneratorsv2")}",
        f"nnUNetv2 @ {get_local_package_path("../segmentation/models/arch/nnunet")}",
        "dynamic_network_architectures",
        "monai",
        "magicgui",
        "magic-class",
        "open3d",
        "numba",
        "s3fs",
        "napari",
        "dask",
        "dask-image",
        "einops",
        "opencv-python-headless",
        "pytorch-lightning",
        "libigl",
        "psutil",
        "pytorch-optimizer",
        "tensorstore",
        "blosc2",
    ],
    python_requires=">=3.8,<3.13",
    include_package_data=True,
    package_data={
        "vesuvius": ["setup/configs/*.yaml"],
        "setup": ["configs/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "vesuvius.accept_terms=setup.accept_terms:main",
            "vesuvius.predict=models.run.inference:main",
            "vesuvius.blend_logits=models.run.blending:main",
            "vesuvius.finalize_outputs=models.run.finalize_outputs:main",
            "vesuvius.inference_pipeline=models.run.vesuvius_pipeline:run_pipeline",
            "vesuvius.compute_st=structure_tensor.run_create_st:main",
            "vesuvius.napari_trainer=utils.napari_trainer.main_window:main",
            "vesuvius.proofreader=utils.vc_proofreader.main:main",
            "vesuvius.voxelize_obj=scripts.voxelize_objs:main",
            "vesuvius.refine_labels=scripts.edt_frangi_label:main",
            "vesuvius.render_obj=rendering.mesh_to_surface:main",
            "vesuvius.flatten_obj=rendering.slim_uv:main",
            "vesuvius.train=models.run.train:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={
        "install": CustomInstallCommand,
    },
)
