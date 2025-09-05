#!/bin/bash

set -e

export CC="ccache clang"
export CXX="ccache clang++"
export INSTALL_PREFIX="$HOME/vc-dependencies"
export BUILD_DIR="$HOME/vc-dependencies-build"
export JOBS=$(nproc)
export COMMON_FLAGS="-march=native -w"
export COMMON_LDFLAGS="-fuse-ld=lld"

if [[ "$(uname)" == "Darwin" ]]; then
    echo "Running on macOS"
    #todo: determine the complete list
    brew install qt cmake clang boost eigen
elif [[ "$(uname)" == "Linux" ]]; then
    echo "Running on Linux"
    sudo apt-get update
    #todo: determine the complete list
    sudo apt-get install -y libgmp-dev libmpfr-dev ccache ninja-build lld \
        libcurl4-openssl-dev libboost-system-dev libboost-program-options-dev qt6-base-dev clang llvm
fi



rm -rf "$BUILD_DIR" "$INSTALL_PREFIX"
mkdir -p "$BUILD_DIR" "$INSTALL_PREFIX"
cd "$BUILD_DIR"

# nlohmann/json
rm -rf json
git clone --depth 1 https://github.com/nlohmann/json.git
cd json
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DJSON_BuildTests=OFF
ninja -j$JOBS
ninja install
cd "$BUILD_DIR"

# xtl
rm -rf xtl
git clone --depth 1 https://github.com/xtensor-stack/xtl.git
cd xtl
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DBUILD_TESTS=OFF
ninja -j$JOBS
ninja install
cd "$BUILD_DIR"

# xsimd
rm -rf xsimd
git clone --depth 1 https://github.com/xtensor-stack/xsimd.git
cd xsimd
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DENABLE_XTL_COMPLEX=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DXSIMD_SKIP_INSTALL=OFF
ninja -j$JOBS
ninja install
cd "$BUILD_DIR"

# xtensor
rm -rf xtensor
git clone --depth 1 https://github.com/xtensor-stack/xtensor.git
cd xtensor
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DBUILD_TESTS=OFF \
    -DXTENSOR_BUILD_TESTS=OFF
ninja -j$JOBS
ninja install
cd "$BUILD_DIR"

# c-blosc
rm -rf c-blosc
git clone --depth 1 https://github.com/Blosc/c-blosc.git
cd c-blosc
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DBUILD_STATIC=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF
ninja -j$JOBS
ninja install
cd "$BUILD_DIR"

# z5
rm -rf z5
git clone --depth 1 https://github.com/SuperOptimizer/z5.git
cd z5
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
    -DWITH_BLOSC=ON \
    -DWITH_ZLIB=ON \
    -DBUILD_Z5PY=OFF \
    -DBUILD_TESTS=OFF
ninja -j$JOBS
ninja install
cd "$BUILD_DIR"

# spdlog
rm -rf spdlog
git clone --depth 1 https://github.com/gabime/spdlog.git
cd spdlog
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DSPDLOG_BUILD_EXAMPLE=OFF \
    -DSPDLOG_BUILD_TESTS=OFF \
    -DSPDLOG_INSTALL=ON \
    -DSPDLOG_FMT_EXTERNAL=OFF
ninja -j$JOBS
ninja install
cd "$BUILD_DIR"

# OpenCV
rm -rf opencv
git clone --depth 1 https://github.com/opencv/opencv.git
cd opencv
mkdir -p build && cd build
cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENMP=OFF \
    -DWITH_TBB=OFF \
    -DWITH_IPP=OFF \
    -DWITH_VTK=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_GSTREAMER=OFF \
    -DWITH_V4L=ON \
    -DWITH_QT=OFF \
    -DWITH_GTK=OFF \
    -DWITH_OPENCL=ON \
    -DWITH_OPENEXR=OFF \
    -DOPENCV_ENABLE_NONFREE=ON
ninja -j$JOBS
ninja install
cd "$BUILD_DIR"

# Ceres Solver
rm -rf ceres-solver
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
mkdir -p ceresbuild && cd ceresbuild

cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_FLAGS="${COMMON_FLAGS} -g0" \
    -DCMAKE_CXX_FLAGS="${COMMON_FLAGS} -g0" \
    -DCMAKE_EXE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${COMMON_LDFLAGS}" \
    -DBUILD_TESTING=OFF \
    -DEIGEN_BUILD_DOC=OFF \
    -DSUITESPARSE=ON \
    -DLAPACK=ON \
    -DEIGENSPARSE=ON \
    -DEIGENMETIS=ON \
    -DSCHUR_SPECIALIZATIONS=ON \
    -DMINIGLOG=ON \
    -DUSE_CUDA=OFF

ninja -j$JOBS
ninja install

echo "Dependencies built successfully!"
echo "Install prefix: $INSTALL_PREFIX"
echo ""
echo "To use these dependencies when building volume-cartographer:"
echo "cd volume-cartographer"
echo "mkdir build"
echo "cd build"
echo "cmake .. -DCMAKE_PREFIX_PATH=~/vc-dependencies -DVC_BUILD_Z5=off -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DVC_USE_OPENMP=off -DVC_INSTALL_APPS=off -DVC_INSTALL_UTILS=off -DVC_INSTALL_LIBS=off"