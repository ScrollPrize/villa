import os
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import warnings
import sys
import subprocess
import platform

version = os.environ.get("VERSION", "0.1.10")


class PostInstallMixin:
    """Mixin for post-installation tasks"""
    def build_betti_matching(self):
        """Clone and build Betti-Matching-3D"""
        # Determine where to put the Betti module
        vesuvius_root = Path(__file__).parent.resolve()
        betti_dir = vesuvius_root / "external" / "Betti-Matching-3D"
        
        try:
            # Create external directory if it doesn't exist
            betti_dir.parent.mkdir(exist_ok=True)
            
            # Clone if not exists
            if not betti_dir.exists():
                print("\nCloning Betti-Matching-3D...")
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/nstucki/Betti-Matching-3D.git",
                    str(betti_dir)
                ], check=True)
            
            # Build
            build_dir = betti_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            print("\nBuilding Betti-Matching-3D...")
            
            # Configure with CMake
            cmake_cmd = ["cmake", ".."]
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                cmake_cmd = ["cmake", "-DCMAKE_OSX_ARCHITECTURES=arm64", ".."]
            
            subprocess.run(cmake_cmd, cwd=build_dir, check=True)
            
            # Build
            subprocess.run(["make"], cwd=build_dir, check=True)
            
            print("\nBetti-Matching-3D built successfully!")
            
        except subprocess.CalledProcessError as e:
            warnings.warn(
                f"\nFailed to build Betti-Matching-3D: {e}\n"
                f"You may need to build it manually:\n"
                f"  cd {betti_dir}\n"
                f"  mkdir build && cd build\n"
                f"  cmake ..\n"
                f"  make\n",
                UserWarning
            )
    
    def post_install(self):
        """Run post-installation tasks"""
        # Build Betti-Matching-3D
        self.build_betti_matching()
        
        message = """
        ============================================================
        Thank you for installing vesuvius!

        To complete the setup, please run the following command:

            vesuvius.accept_terms --yes

        This will display the terms and conditions to be accepted.
        ============================================================
        """
        warnings.warn(message, UserWarning)


class CustomInstallCommand(PostInstallMixin, install):
    def run(self):
        install.run(self)
        self.post_install()


class CustomDevelopCommand(PostInstallMixin, develop):
    def run(self):
        develop.run(self)
        self.post_install()


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
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url="https://github.com/ScrollPrize/villa",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "pybind11>=2.5.0",
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
        "torch>=2.6",
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
        "wandb",
        "wandb[media]"
    ],
    python_requires=">=3.8,<3.13",
    include_package_data=True,
    package_data={
        "vesuvius": ["setup/configs/*.yaml"],
        "setup": ["configs/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "vesuvius.accept_terms=vesuvius.setup.accept_terms:main",
            "vesuvius.predict=vesuvius.models.run.inference:main",
            "vesuvius.blend_logits=vesuvius.models.run.blending:main",
            "vesuvius.finalize_outputs=vesuvius.models.run.finalize_outputs:main",
            "vesuvius.inference_pipeline=vesuvius.models.run.vesuvius_pipeline:run_pipeline",
            "vesuvius.compute_st=vesuvius.structure_tensor.run_create_st:main",
            "vesuvius.napari_trainer=vesuvius.utils.napari_trainer.main_window:main",
            "vesuvius.voxelize_obj=vesuvius.scripts.voxelize_objs:main",
            "vesuvius.refine_labels=vesuvius.scripts.edt_frangi_label:main",
            "vesuvius.render_obj=vesuvius.rendering.mesh_to_surface:main",
            "vesuvius.flatten_obj=vesuvius.rendering.slim_uv:main",
            "vesuvius.train=vesuvius.models.training.train:main",
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
        "develop": CustomDevelopCommand,
    },
)
