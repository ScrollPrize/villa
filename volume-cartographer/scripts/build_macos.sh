#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/build_macos.sh [--install-deps] [--build-dir DIR] [--jobs N]

Build VC3D natively on macOS with Homebrew LLVM/Clang.

Options:
  --install-deps   Install missing Homebrew formulae before configuring.
  --build-dir DIR  Override the default build directory: build-macos
  --jobs N         Parallel build jobs. Defaults to the host CPU count.
USAGE
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS only." >&2
  exit 1
fi

# Run from the volume-cartographer source root so `cmake --preset` finds
# CMakePresets.json regardless of caller cwd.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

install_deps=0
build_dir="build-macos"
jobs="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-deps)
      install_deps=1
      shift
      ;;
    --build-dir)
      build_dir="${2:?missing value for --build-dir}"
      shift 2
      ;;
    --jobs)
      jobs="${2:?missing value for --jobs}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh/ and rerun this script." >&2
  exit 1
fi

brew_prefix="$(brew --prefix)"
export HOMEBREW_PREFIX="$brew_prefix"
required_formulae=(
  llvm
  # As of Homebrew llvm 19+, ld64.lld no longer ships in the llvm formula;
  # the Mach-O backend lives in a standalone `lld` keg.
  lld
  # LLVM's Fortran frontend; lives in its own keg (Homebrew's llvm formula
  # deliberately doesn't ship flang). PaStiX enables Fortran for a tiny
  # configure-time check; flang+lld speaks the Mach-O linker correctly
  # whereas gfortran does not.
  flang
  cmake
  ninja
  pkgconf
  qt
  ceres-solver
  eigen
  opencv
  cgal
  c-blosc
  boost
  libtiff
  curl
  nlohmann-json
  lapack
  # OpenBLAS provides cblas_* with the legacy ILP32 symbol names PaStiX
  # expects. Accelerate.framework's BLAS hides those symbols when linked
  # with -fuse-ld=lld + thin LTO, so we force PaStiX onto OpenBLAS instead.
  openblas
  scotch
  hwloc
  # gcc stays in the deps list: OpenBLAS/lapack carry a libgfortran runtime
  # dependency even though we no longer use gfortran as the compiler.
  gcc
)

missing=()
for formula in "${required_formulae[@]}"; do
  if ! brew list --versions "$formula" >/dev/null 2>&1; then
    missing+=("$formula")
  fi
done

if (( ${#missing[@]} > 0 )); then
  if (( install_deps )); then
    brew install "${missing[@]}"
  else
    echo "Missing Homebrew formulae: ${missing[*]}" >&2
    echo "Install them with:" >&2
    echo "  brew install ${missing[*]}" >&2
    echo "or rerun this script with --install-deps." >&2
    exit 1
  fi
fi

llvm_bin="$brew_prefix/opt/llvm/bin"
lld_bin="$brew_prefix/opt/lld/bin"
if [[ ! -x "$llvm_bin/clang++" ]]; then
  echo "Homebrew LLVM compiler not found at $llvm_bin/clang++." >&2
  exit 1
fi
# Homebrew llvm 19+ ships lld in a standalone keg (`brew install lld`).
# The preset's -fuse-ld=lld asks clang's driver for ld64.lld; we need it
# resolvable on PATH.
if [[ ! -x "$lld_bin/ld64.lld" ]]; then
  echo "Homebrew LLD (ld64.lld) not found at $lld_bin/ld64.lld." >&2
  echo "Install with: brew install lld" >&2
  exit 1
fi
# Put lld_bin and llvm_bin first so clang's driver finds ld64.lld plus
# llvm-strip / llvm-nm / llvm-install-name-tool without falling back to
# Apple's /usr/bin/*. Order: lld_bin before llvm_bin (lld_bin only has
# the linker binaries, doesn't shadow anything else).
export PATH="$lld_bin:$llvm_bin:$PATH"

# Homebrew LLVM's clang++ needs SDKROOT pointing at the active Xcode/CLT SDK
# so its libc++ headers compose correctly with the C SDK headers (otherwise
# Xcode 16's _string.h sees `size_t` in std:: only and fails to compile).
# SDKROOT is the *only* Apple-toolchain dependency we keep — there is no
# Homebrew replacement for the macOS system headers or libSystem.
if command -v xcrun >/dev/null 2>&1; then
  export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"
fi

extra_cmake_args=()
# Eigen ships its CMake config under <prefix>/share/eigen3/cmake. Derive
# the prefix via `brew --prefix eigen` so a Homebrew revision bump (eigen
# 3.4.0_1 → 3.4.0_2 → 3.5.x …) doesn't rot the script.
eigen_prefix="$(brew --prefix eigen 2>/dev/null || true)"
if [[ -n "$eigen_prefix" && -d "$eigen_prefix/share/eigen3/cmake" ]]; then
  extra_cmake_args+=("-DEigen3_DIR=$eigen_prefix/share/eigen3/cmake")
fi
if [[ -d "$brew_prefix/opt/nlohmann-json/share/cmake/nlohmann_json" ]]; then
  extra_cmake_args+=("-Dnlohmann_json_DIR=$brew_prefix/opt/nlohmann-json/share/cmake/nlohmann_json")
fi

# lapack and openblas are keg-only on macOS (Accelerate ships LAPACK but not
# LAPACKE, which PaStiX requires; openblas would otherwise be masked by
# Accelerate). Make both discoverable by pkg-config and cmake's find_package.
for keg in lapack openblas; do
  if [[ -d "$brew_prefix/opt/$keg" ]]; then
    export PKG_CONFIG_PATH="$brew_prefix/opt/$keg/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export CMAKE_PREFIX_PATH="$brew_prefix/opt/$keg${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  fi
done
# Force PaStiX onto OpenBLAS instead of letting morse_cmake fall back to
# Accelerate.framework — see openblas note in required_formulae above.
extra_cmake_args+=("-DBLA_VENDOR=OpenBLAS")

# PaStiX enables Fortran at configure time even with PASTIX_WITH_FORTRAN=OFF.
# Use LLVM's flang (from the Homebrew flang keg) so the toolchain stays
# clang/flang + lld end-to-end; gfortran would inherit our -fuse-ld=lld
# linker flags and ld64.lld would reject the link (gfortran doesn't pass
# -platform_version / -arch).
flang_bin="$(brew --prefix flang 2>/dev/null || true)/bin/flang"
if [[ -x "$flang_bin" ]]; then
  export FC="$flang_bin"
  extra_cmake_args+=("-DCMAKE_Fortran_COMPILER=$flang_bin")
else
  echo "Homebrew flang not found at $flang_bin (run with --install-deps)" >&2
  exit 1
fi

cmake --preset macos-homebrew-llvm \
  -B "$build_dir" \
  "${extra_cmake_args[@]}"

cmake --build "$build_dir" -j "$jobs"
