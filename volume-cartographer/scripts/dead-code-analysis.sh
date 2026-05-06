#!/usr/bin/env bash
# Dead-code analysis using nm (approach B):
#   dead = (defined symbols in our source .o/.a files) - (symbols present in any final binary)
#
# Runs inside the builder image (binutils + c++filt available).
# Reads cmake build dir from $1, writes report files under dead-code/.

set -euo pipefail

build_dir=${1:?usage: $0 <cmake-build-dir>}
out=dead-code
mkdir -p "$out"

# Compile-time -Wunused* / -Wunreachable* warnings (assumes build.log is in $out).
if [[ -f "$out/build.log" ]]; then
    grep -E '\[-W(unused|unreachable)' "$out/build.log" \
        | sort -u > "$out/compile-warnings.txt" || true
else
    : > "$out/compile-warnings.txt"
fi

# Defined-symbol filter: keep T/t (text), D/d (data), B/b (bss), R/r (rodata),
# W/w (weak), V/v (weak object). Locals (lowercase) survive in .o files but
# vanish from binaries — that's exactly the diff we want, so include both.
nm_filter='$2 ~ /^[TtDdBbRrWwVv]$/ { print $NF }'

# Our source: .o files under CMakeFiles, excluding autogen/_deps/external.
mapfile -t src_objs < <(find \
    "$build_dir/core" "$build_dir/utils" "$build_dir/apps" "$build_dir/libs/c3d" \
    -path '*/CMakeFiles/*' -name '*.o' ! -path '*_autogen*' 2>/dev/null)

# Final binaries.
mapfile -t binaries < <(find "$build_dir/bin" -maxdepth 1 -type f -executable 2>/dev/null)

if (( ${#src_objs[@]} == 0 )); then
    echo "no source .o files under $build_dir — bailing" >&2
    exit 1
fi
if (( ${#binaries[@]} == 0 )); then
    echo "no executables under $build_dir/bin — bailing" >&2
    exit 1
fi

nm --defined-only --no-sort "${src_objs[@]}" 2>/dev/null \
    | awk "$nm_filter" | sort -u > "$out/source-syms.txt"
nm --defined-only --no-sort "${binaries[@]}" 2>/dev/null \
    | awk "$nm_filter" | sort -u > "$out/binary-syms.txt"

comm -23 "$out/source-syms.txt" "$out/binary-syms.txt" \
    | c++filt > "$out/dead-symbols-raw.txt"

# Strip noise: exception tables, DWARF refs, RTTI bits, static-init guards,
# compiler-internal labels — these are emitted by the compiler/runtime, not
# user-authored "dead code".
grep -vE '^(GCC_except_table|DW\.ref\.|__|\.L|guard variable for |vtable for |typeinfo (name )?for |construction vtable for |VTT for |non-virtual thunk |virtual thunk )' \
    "$out/dead-symbols-raw.txt" > "$out/dead-symbols.txt" || true

n_src=$(wc -l < "$out/source-syms.txt")
n_bin=$(wc -l < "$out/binary-syms.txt")
n_dead_raw=$(wc -l < "$out/dead-symbols-raw.txt")
n_dead=$(wc -l < "$out/dead-symbols.txt")
n_warn=$(wc -l < "$out/compile-warnings.txt")

{
    echo "Dead-code report (build=$build_dir)"
    echo "  Source-defined symbols (.o files):                   $n_src"
    echo "  Symbols present in any final binary:                 $n_bin"
    echo "  Defined-but-not-in-any-binary, raw:                  $n_dead_raw"
    echo "  Defined-but-not-in-any-binary, after noise filter:   $n_dead"
    echo "  Compile-time -Wunused* / -Wunreachable* hits:        $n_warn"
    echo
    echo "Top 50 dead symbols (demangled, noise-filtered):"
    head -50 "$out/dead-symbols.txt"
    echo
    echo "Top 30 compile warnings:"
    head -30 "$out/compile-warnings.txt"
} | tee "$out/summary.txt"
