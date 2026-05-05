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
if [[ ! -x "$llvm_bin/clang++" ]]; then
  echo "Homebrew LLVM compiler not found at $llvm_bin/clang++." >&2
  exit 1
fi

extra_cmake_args=()
if [[ -d "$brew_prefix/Cellar/eigen/3.4.0_1/share/eigen3/cmake" ]]; then
  extra_cmake_args+=("-DEigen3_DIR=$brew_prefix/Cellar/eigen/3.4.0_1/share/eigen3/cmake")
fi
if [[ -d "$brew_prefix/opt/nlohmann-json/share/cmake/nlohmann_json" ]]; then
  extra_cmake_args+=("-Dnlohmann_json_DIR=$brew_prefix/opt/nlohmann-json/share/cmake/nlohmann_json")
fi

cmake --preset macos-homebrew-llvm \
  -B "$build_dir" \
  "${extra_cmake_args[@]}"

cmake --build "$build_dir" -j "$jobs"
