#!/usr/bin/env bash
# Build OpenCV 4.12.0's core and imgproc as static libraries (no Eigen, no codecs) for the vc_reader, vc_object_store
# and spiral wheels, so they bundle no OpenCV dylibs. Without Eigen, because a static install exports Homebrew's
# Eigen3::Eigen in opencv_core's link interface, which the modules' find_package(OpenCV) cannot resolve
#   build-opencv.sh PREFIX [JOBS]   honours MACOSX_DEPLOYMENT_TARGET
set -euo pipefail
prefix=$1; jobs=${2:-4}
# the GitHub tag archive as of Nov 25 2026
sha=44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d
work=$(mktemp -d)
curl -fsSL -o "$work/opencv.tar.gz" https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz
[ "$(shasum -a 256 "$work/opencv.tar.gz" | cut -d' ' -f1)" = "$sha" ] || { echo "OpenCV archive checksum mismatch" >&2; exit 1; }
tar -xzf "$work/opencv.tar.gz" -C "$work"
cmake -S "$work/opencv-4.12.0" -B "$work/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DBUILD_LIST=core,imgproc -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DBUILD_ZLIB=ON -DWITH_OPENEXR=OFF \
  -DWITH_TIFF=OFF -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_AVIF=OFF \
  -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF -DWITH_PROTOBUF=OFF -DBUILD_PROTOBUF=OFF -DWITH_ADE=OFF \
  -DBUILD_opencv_dnn=OFF -DBUILD_opencv_gapi=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DWITH_EIGEN=OFF
cmake --build "$work/build" -j "$jobs"
cmake --install "$work/build"
rm -rf "$work"
