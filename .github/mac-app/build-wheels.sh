#!/usr/bin/env bash
# Build the villa native Python packages as self-contained cp314 macOS arm64 wheels, so the Python tools installer never
# compiles anything:
#   vc-reader, vc-object-store    CPU builds against a static OpenCV core (vc_object_store without its server)
#   spiral                        spiral-fitting's vc_spiral module
#   volume-cartographer           the vc bindings Lasagna uses (libvc_core, libvc_lasagna, fiber_trace, volume)
#   vesuvius-lasagna              Lasagna with its two pybind11 modules
#   villa-gpu-manager             pure Python
#   posix-ipc 1.3.2               third party: spiral-fitting pins it, and PyPI has no cp314 macOS wheel for it
# Each wheel is built with uv's managed CPython (the installer's), delocated (Homebrew dylibs copied in, except libomp),
# then fix_wheel.py points libomp at torch's copy and checks that no path leads back to Homebrew or a build directory.
#   build-wheels.sh SRC OUT [JOBS]   SRC: a villa tree (a git archive of the checkout), OUT: wheel output directory
# Environment:
#   MACOSX_DEPLOYMENT_TARGET  the code's macOS floor and the wheels' tag (default 27.0). Tags only go by major version
#                             from macOS 11 on (macosx_N_0)
#   BUNDLED_LIBS_TARGET       the newest macOS any bundled Homebrew library is built for (default: the deployment
#                             target). On a Homebrew built for a point release above it, such as 27.2, set that here:
#                             the tag still says N_0, so the installer has to check the point release itself
#   OCV                       a static OpenCV core and imgproc install (.github/mac-app/build-opencv.sh)
#   TORCH_LIBOMPS             space-separated torch/lib/libomp.dylib files the wheels must load (every torch version
#                             the locks pin)
#   REUSE_RAW=1               skip the builds and post-process the wheels already in OUT/raw
set -euo pipefail
src=$(cd "$1" && pwd); out=$2; jobs=${3:-6}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
BREW=$(brew --prefix)
PYV=3.14.10
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-27.0}
BUNDLED_LIBS_TARGET=${BUNDLED_LIBS_TARGET:-$MACOSX_DEPLOYMENT_TARGET}
# a static OpenCV core and imgproc, so vc_reader and vc_object_store bundle no OpenCV dylibs
OCV=${OCV:?set OCV to a static OpenCV install (build-opencv.sh)}
[ -f "$OCV/lib/cmake/opencv4/OpenCVConfig.cmake" ] || { echo "no OpenCV install at $OCV" >&2; exit 1; }
read -r -a TORCH_LIBOMPS <<< "${TORCH_LIBOMPS:?set TORCH_LIBOMPS to the torch libomp.dylib files to load}"

uv python install "$PYV"
PY=$(uv python find --python-preference only-managed "$PYV")
echo "building with $PY ($("$PY" -V)) for macOS $MACOSX_DEPLOYMENT_TARGET (bundled libraries $BUNDLED_LIBS_TARGET) at $(date)"

# Apple clang: Homebrew's LLVM fails against the SDK's libc++ headers; it has no OpenMP, so FindOpenMP gets libomp
export CC=/usr/bin/clang CXX=/usr/bin/clang++ OpenMP_ROOT=$BREW/opt/libomp
export CMAKE_BUILD_PARALLEL_LEVEL=$jobs
export PKG_CONFIG_PATH=$BREW/opt/zstd/lib/pkgconfig:$BREW/opt/sqlite/lib/pkgconfig:$BREW/lib/pkgconfig
raw=$out/raw; del=$out/delocated; fin=$out/wheels; bld=$out/build
rm -rf "$del" "$fin"; mkdir -p "$raw" "$del" "$fin" "$bld"

build() {
  local name=$1 dir=$2; shift 2
  echo "== $name at $(date '+%-I:%M %p')"
  uv build --wheel --python "$PY" --out-dir "$raw" "$@" "$dir"
}

if [ "${REUSE_RAW:-0}" != 1 ]; then
rm -f "$raw"/*.whl
# vc_object_store and vc_reader: Homebrew's SQLite, not Apple's (Apple's keeps WAL files after close, which
# fiber_store refuses); the store embeds its own static vc_reader
CMAKE_PREFIX_PATH="$OCV;$BREW/opt/sqlite;$BREW/opt/openssl@3;$BREW/opt/nlohmann-json;$BREW/opt/zstd;$BREW" \
  build vc-reader "$src/vc_reader" -C cmake.define.VC_READER_ENABLE_CUDA=OFF -C "build-dir=$bld/vc_reader" \
  -C "cmake.define.OpenCV_DIR=$OCV/lib/cmake/opencv4" -C cmake.define.SQLite3_ROOT=$BREW/opt/sqlite
CMAKE_PREFIX_PATH="$OCV;$BREW/opt/sqlite;$BREW/opt/openssl@3;$BREW/opt/nlohmann-json;$BREW/opt/zstd;$BREW" \
  build vc-object-store "$src/vc_object_store" -C cmake.define.VC_READER_ENABLE_CUDA=OFF \
  -C cmake.define.VC_OBJECT_STORE_BUILD_SERVER=OFF -C cmake.define.SQLite3_ROOT=$BREW/opt/sqlite \
  -C cmake.define.CMAKE_DISABLE_FIND_PACKAGE_vc_reader=ON -C "cmake.define.OpenCV_DIR=$OCV/lib/cmake/opencv4" \
  -C "build-dir=$bld/vc_object_store"
CMAKE_PREFIX_PATH="$OCV;$BREW" build spiral "$src/spiral-fitting" -C "build-dir=$bld/spiral" \
  -C "cmake.define.OpenCV_DIR=$OCV/lib/cmake/opencv4"
# the vc bindings need the full OpenCV (calib, features, imgcodecs, ...), Ceres and Eigen: Homebrew's
CMAKE_PREFIX_PATH="$BREW/opt/sqlite;$BREW/opt/openssl@3;$BREW" \
  build volume-cartographer "$src/volume-cartographer" -C "build-dir=$bld/volume-cartographer" \
  -C cmake.define.VC_READER_ENABLE_CUDA=OFF -C cmake.define.SQLite3_ROOT=$BREW/opt/sqlite
build vesuvius-lasagna "$src/lasagna"
build villa-gpu-manager "$src/gpu_manager"
# posix-ipc 1.3.2's sdist, pinned to the sha256 PyPI listed when this script was written
name=posix_ipc-1.3.2.tar.gz
sha=6923232111329954a8349f7d99f212b6e96b5206e77fbd39aaf1b3cb4a5e9260
url=$(curl -fsSL https://pypi.org/pypi/posix-ipc/1.3.2/json | "$PY" -I -c '
import json, sys
print(next(u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"] == "sdist"))')
curl -fsSL -o "$bld/$name" "$url"
[ "$(shasum -a 256 "$bld/$name" | cut -d' ' -f1)" = "$sha" ] || { echo "$name checksum mismatch" >&2; exit 1; }
build posix-ipc "$bld/$name"
fi
ls -la "$raw"

stripped=$out/stripped
rm -rf "$stripped"; mkdir -p "$stripped"
for whl in "$raw"/*.whl; do
  keep=; case $(basename "$whl") in vc_reader-*) keep=--keep-lib-dylibs ;; esac
  "$PY" -I "$here/fix_wheel.py" strip "$whl" "$stripped" $keep
done
for whl in "$stripped"/*.whl; do
  case $(basename "$whl") in
    *-none-any.whl) cp "$whl" "$fin/" ;;
    *) MACOSX_DEPLOYMENT_TARGET=$BUNDLED_LIBS_TARGET uvx --from delocate==0.13.0 delocate-wheel --require-archs arm64 --exclude libomp -w "$del" -v "$whl" ;;
  esac
done
for whl in "$del"/*.whl; do
  "$PY" -I "$here/fix_wheel.py" fix "$whl" "$fin" "${TORCH_LIBOMPS[@]}"
done
"$PY" -I "$here/fix_wheel.py" check "$fin"/*.whl
(cd "$fin" && shasum -a 256 ./*.whl | sed 's| \./| |' > SHA256SUMS)
ls -la "$fin"; cat "$fin/SHA256SUMS"
echo "wheels ready in $fin at $(date)"
