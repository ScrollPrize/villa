# PGS Recon
A Python-based pipeline for reconstructing photogrammetry datasets. Includes a CMake project for compiling the 
required executables.

## Installation
### Requirements
The Python script uses executables provided by the OpenMVG and OpenMVS projects. The included CMake project will 
compile both of these projects as well as their dependencies. Before configuring the CMake project, please preinstall 
the following dependencies:
* CMake 3.16+
* Boost 1.70+
* GMP and MPFR
* ExifTool
* (Optional) NASM (Required by jpeg-turbo)
* (Optional) Ceres Solver
* (Optional) CUDA Toolkit

After the dependencies have been installed, configure and build the CMake project to compile the required executables:
```shell
cmake -S dependencies -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/
```

**Notes**:
* OpenMVS requires a specific version of VCG (built by this project). This CMake script makes every attempt to use the 
  locally built VCG when building OpenMVS, however, other versions of VCG in the system path may cause conflicts. If 
  you have build issues with OpenMVS, you may temporarily remove these installations.
* OpenMVS and OpenCV must be linked against the same version of libjpeg.

### Pipelines
Use a recent version of `pip` to install the Python scripts:
```shell
# Requires Python 3.9+
python3 -m pip install .
```

After installation, the reconstruction script can be run from the shell:
```shell
pgs-recon --help
```

### Advanced Installation
#### Installation Location
By default, executables created by this CMake project will be installed to `installed/`. The installation location can 
be changed by setting the CMake installation prefix flag:
```shell
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ ..
```

#### Disable compilation of extra libraries
In addition to VCG, OpenMVG, and OpenMVS, the CMake project also compiles a number of required software libraries. We 
provide corresponding CMake flags to control the compilation of these libraries. To use a system-provided version of 
these libraries, set the library's flag to `OFF`:

```cmake
BUILD_EIGEN: If ON, builds Eigen 3.2
BUILD_JPEG: If ON, builds libjpeg
BUILD_JPEG_TURBO: If ON, builds libjpeg-turbo (depends BUILD_JPEG=ON)
BUILD_OPENCV: If ON, builds OpenCV
BUILD_CGAL: If ON, builds CGAL
```
