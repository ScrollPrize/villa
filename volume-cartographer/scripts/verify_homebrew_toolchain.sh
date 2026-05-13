#!/usr/bin/env bash
# CI assertion: the macOS build under <build_dir>/bin/ used only Homebrew
# LLVM tools and lld. Apple's toolchain is allowed to contribute exactly two
# things and nothing else:
#   1. SDKROOT — the system headers/frameworks. There is no alternative.
#   2. /usr/lib/libSystem.dylib (and its dependencies) — the syscall ABI;
#      every Mach-O binary links it.
#
# Anything else from /Applications/Xcode.app/ or /Library/Developer/ leaking
# into the binaries means the toolchain swap regressed and Apple clang/ld
# crept back in. Fail loudly so we can catch it in review.

set -euo pipefail

build_dir=${1:?usage: $0 <macos-build-dir>}

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "$0: must run on macOS (uname=$(uname -s))" >&2
    exit 2
fi

bins=(VC3D vc_tifxyz flatboi)
fail=0

echo "== Build-info banners =="
# clang --version output gets baked into __TEXT,__compiler_used /
# LC_BUILD_VERSION on macOS. Easier: check the compile_commands.json that
# the preset emits to confirm the configure step used Homebrew clang.
ccdb="$build_dir/compile_commands.json"
if [[ -f "$ccdb" ]]; then
    if ! grep -q "/opt/homebrew/opt/llvm/bin/clang\|/usr/local/opt/llvm/bin/clang" "$ccdb"; then
        echo "FAIL: compile_commands.json does not reference Homebrew LLVM clang" >&2
        head -1 "$ccdb" | tr ',' '\n' | head -20 >&2
        fail=1
    else
        echo "ok: compile_commands.json references Homebrew LLVM clang"
    fi
else
    echo "warn: $ccdb missing — skipping compiler-path check" >&2
fi

echo
echo "== Dynamic dependencies (otool -L, allowlist) =="
# Allowed dylib prefixes — anything not on this list is treated as a
# toolchain leak. otool -L reports direct deps only; transitive deps could
# in principle still pull in Xcode-internal frameworks, but our final
# binaries should not transitively depend on anything Apple-developer-tools.
allowed_prefixes='^(\s+)?(/usr/lib/|/System/Library/Frameworks/|/System/Library/PrivateFrameworks/|/opt/homebrew/|/usr/local/|@rpath/|@loader_path/|@executable_path/)'

for b in "${bins[@]}"; do
    bin_path="$build_dir/bin/$b"
    if [[ ! -x "$bin_path" ]]; then
        echo "skip: $b not built"
        continue
    fi
    # `otool -L` first line is the binary path itself; skip it. Each
    # remaining line is "<path> (compatibility version …)" — match the path.
    bad=$(otool -L "$bin_path" | tail -n +2 | grep -vE "$allowed_prefixes" || true)
    if [[ -n "$bad" ]]; then
        echo "FAIL: $b depends on dylibs outside the allowlist:" >&2
        echo "$bad" >&2
        fail=1
    else
        echo "ok: $b dylib deps are all on the allowlist"
    fi
done

echo
echo "== libc++ origin =="
# Homebrew clang links its own libc++ from $HOMEBREW_PREFIX/opt/llvm/lib
# (often via @rpath/libc++.1.dylib or similar). Apple clang would emit
# /usr/lib/libc++.1.dylib directly. Match Homebrew's actual SONAME variants
# (libc++.1.dylib, libc++.1.0.dylib, etc.). If we see /usr/lib/libc++ the
# build slipped onto Apple clang.
for b in "${bins[@]}"; do
    bin_path="$build_dir/bin/$b"
    [[ -x "$bin_path" ]] || continue
    cxx_line=$(otool -L "$bin_path" | grep -E 'libc\+\+\.[0-9.]+\.dylib' || true)
    if [[ -z "$cxx_line" ]]; then
        continue  # Static libc++, also fine.
    fi
    if echo "$cxx_line" | grep -qE '^\s*/usr/lib/libc\+\+'; then
        echo "FAIL: $b links Apple's /usr/lib/libc++ (expected Homebrew LLVM libc++):" >&2
        echo "  $cxx_line" >&2
        fail=1
    else
        echo "ok: $b uses non-Apple libc++ ($(echo "$cxx_line" | awk '{print $1}'))"
    fi
done

if (( fail )); then
    echo
    echo "verify_homebrew_toolchain: FAILED — Apple toolchain leaked into the build" >&2
    exit 1
fi
echo
echo "verify_homebrew_toolchain: OK"
