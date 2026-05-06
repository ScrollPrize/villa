#!/usr/bin/env bash
# Dead-code analysis using nm + ninja depfiles. Five reports:
#
#   1. dead-symbols.txt   — symbols defined in our .o files but absent
#                           from every final binary (after a noise filter
#                           for compiler-emitted boilerplate).
#   2. dead-objects.txt   — .cpp files that compiled to .o but contributed
#                           zero defined symbols to any final binary.
#                           Whole TU is unreachable.
#   3. uncompiled-cpps.txt— .cpp files in the source tree that the build
#                           didn't compile. Any symbol referenced only
#                           from such a file would show up as a false
#                           positive in dead-symbols, so this is the
#                           soundness check for that report.
#   4. dead-headers.txt   — .h / .hpp files in the source tree that no
#                           compiled .cpp transitively #include's,
#                           determined from ninja .d depfiles.
#   5. compile-warnings.txt — -Wunused-* / -Wunreachable-* compile hits.
#
# Runs inside the builder image (binutils + c++filt available).

set -euo pipefail

build_dir=${1:?usage: $0 <cmake-build-dir>}
out=dead-code
mkdir -p "$out"

# Source roots we own. Anything outside (e.g. _deps, libs/libigl_changes,
# vendored externals) is excluded from the "source" sets — we don't
# care if vendored code has unused symbols.
src_roots=(core utils apps libs/c3d libs/OpenABF/include)
src_roots_existing=()
for r in "${src_roots[@]}"; do
    [[ -d "$r" ]] && src_roots_existing+=("$r")
done

# ------------------------------------------------------------------
# 5. Compile-time -Wunused* / -Wunreachable* warnings
# ------------------------------------------------------------------
if [[ -f "$out/build.log" ]]; then
    grep -E '\[-W(unused|unreachable)' "$out/build.log" \
        | sort -u > "$out/compile-warnings.txt" || true
else
    : > "$out/compile-warnings.txt"
fi

# ------------------------------------------------------------------
# Collect inputs we need from the build tree
# ------------------------------------------------------------------

# All .cpp files we own
find "${src_roots_existing[@]}" -type f -name '*.cpp' ! -path '*_autogen*' 2>/dev/null \
    | sort -u > "$out/all-cpps.txt"

# All .h / .hpp files we own
find "${src_roots_existing[@]}" -type f \( -name '*.h' -o -name '*.hpp' \) ! -path '*_autogen*' 2>/dev/null \
    | sort -u > "$out/all-headers.txt"

# All .o files the build produced under our targets (skip _autogen, _deps)
find "$build_dir" -path '*/CMakeFiles/*' -name '*.o' \
    ! -path '*_autogen*' ! -path '*/_deps/*' 2>/dev/null \
    | sort -u > "$out/all-objects.txt"

# Ninja's binary depfile database — the canonical place to find what
# headers/sources each .o was compiled from. Per-file .d files don't
# exist with the ninja generator.
( cd "$build_dir" && ninja -t deps 2>/dev/null ) > "$out/ninja-deps.txt"

# Final binaries
find "$build_dir/bin" -maxdepth 1 -type f -executable 2>/dev/null \
    | sort -u > "$out/all-binaries.txt"

# ------------------------------------------------------------------
# 3+4. Parse ninja deps. Format per block:
#        <obj>: #deps N, deps mtime ... (VALID)
#            <dep1>
#            <dep2>
#            ...
# Indented lines are dependencies (sources + headers).
#
# Build:
#   built-cpps.txt   — every .cpp/.c/.cxx/.cc that appears as a dep
#                      of one of our .o targets.
#   included-headers.txt — every .h/.hpp that appears as a dep.
# Both restricted to our source tree (paths under src_roots).
# ------------------------------------------------------------------
awk -v root="$PWD" '
    /^[^[:space:]].*: #deps/ { in_block = 1; next }
    /^$/ { in_block = 0; next }
    in_block && /^[[:space:]]/ {
        sub(/^[[:space:]]+/, "")
        sub("^/src/", "")
        sub("^" root "/", "")
        print
    }
' "$out/ninja-deps.txt" \
    | sort -u > "$out/all-deps.txt"

# Restrict to our source roots
roots_re="^($(IFS='|'; echo "${src_roots_existing[*]}"))/"
grep -E "$roots_re" "$out/all-deps.txt" > "$out/our-deps.txt" || true

grep -E '\.(cpp|cc|cxx|c)$' "$out/our-deps.txt" | sort -u > "$out/built-cpps.txt" || true
grep -E '\.(h|hpp)$'        "$out/our-deps.txt" | sort -u > "$out/included-headers.txt" || true

comm -23 "$out/all-cpps.txt"    "$out/built-cpps.txt"       > "$out/uncompiled-cpps.txt"
comm -23 "$out/all-headers.txt" "$out/included-headers.txt" > "$out/dead-headers.txt"

# ------------------------------------------------------------------
# 1+2. Symbol-level analysis
# ------------------------------------------------------------------
# Defined-symbol filter: T/t/D/d/B/b/R/r/W/w/V/v.
nm_filter='$2 ~ /^[TtDdBbRrWwVv]$/ { print $NF }'

# Symbols defined per source .o (we record symbol → .o owners so we
# can later answer "which .o is fully dead"). Fall back to nm on the
# whole batch for the union-set.
mapfile -t src_objs < "$out/all-objects.txt"
mapfile -t binaries < "$out/all-binaries.txt"
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

comm -23 "$out/source-syms.txt" "$out/binary-syms.txt" > "$out/dead-symbols-mangled.txt"
c++filt < "$out/dead-symbols-mangled.txt" > "$out/dead-symbols-raw.txt"

grep -vE '^(GCC_except_table|DW\.ref\.|__|\.L|guard variable for |vtable for |typeinfo (name )?for |construction vtable for |VTT for |non-virtual thunk |virtual thunk )' \
    "$out/dead-symbols-raw.txt" > "$out/dead-symbols.txt" || true

# Per-.o liveness: an object is "fully dead" iff none of its defined
# symbols appear in any binary. nm with --print-file-name emits each
# symbol as `path:type:name` so we can group by file.
nm --defined-only --print-file-name --no-sort "${src_objs[@]}" 2>/dev/null \
    | awk -v RS='\n' '$2 ~ /^[TtDdBbRrWwVv]$/ {
        n = split($1, a, ":"); file = a[1];
        for (i=2;i<n;i++) file = file ":" a[i];
        sym = $NF;
        print file "\t" sym
    }' > "$out/object-symbols.tsv"

awk -F'\t' 'NR==FNR { live[$1]=1; next } { obj_syms[$1] = obj_syms[$1] "\n" $2 }
    END { for (o in obj_syms) {
            split(obj_syms[o], syms, "\n");
            any_live = 0;
            for (i in syms) if (syms[i] != "" && (syms[i] in live)) { any_live = 1; break }
            if (!any_live) print o
          } }' \
    "$out/binary-syms.txt" "$out/object-symbols.tsv" \
    | sort -u > "$out/dead-objects.txt"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
n_cpp=$(wc -l < "$out/all-cpps.txt")
n_built=$(wc -l < "$out/built-cpps.txt")
n_uncompiled=$(wc -l < "$out/uncompiled-cpps.txt")
n_dead_obj=$(wc -l < "$out/dead-objects.txt")
n_hdr=$(wc -l < "$out/all-headers.txt")
n_dead_hdr=$(wc -l < "$out/dead-headers.txt")
n_src=$(wc -l < "$out/source-syms.txt")
n_bin=$(wc -l < "$out/binary-syms.txt")
n_dead_sym_raw=$(wc -l < "$out/dead-symbols-raw.txt")
n_dead_sym=$(wc -l < "$out/dead-symbols.txt")
n_warn=$(wc -l < "$out/compile-warnings.txt")

{
    echo "Dead-code report (build=$build_dir)"
    echo
    echo "Source files:"
    echo "  .cpp in source tree:                  $n_cpp"
    echo "  .cpp compiled by the build:           $n_built"
    echo "  .cpp NOT compiled (false-pos risk):   $n_uncompiled"
    echo "  .cpp compiled but linked-into-zero:   $n_dead_obj"
    echo
    echo "Headers:"
    echo "  .h/.hpp in source tree:               $n_hdr"
    echo "  .h/.hpp not included anywhere:        $n_dead_hdr"
    echo
    echo "Symbols (only sound if uncompiled .cpp count above is 0):"
    echo "  Defined in source .o files:           $n_src"
    echo "  Present in any final binary:          $n_bin"
    echo "  Dead (raw):                           $n_dead_sym_raw"
    echo "  Dead (after compiler-noise filter):   $n_dead_sym"
    echo
    echo "Compile-time -Wunused* / -Wunreachable* warnings: $n_warn"
    echo
    if (( n_uncompiled > 0 )); then
        echo "Uncompiled .cpp files (sources of false positives below):"
        cat "$out/uncompiled-cpps.txt" | sed 's/^/  /'
        echo
    fi
    echo "Top 30 dead .cpp files (entire TU contributes nothing):"
    head -30 "$out/dead-objects.txt" | sed 's|.*/CMakeFiles/||; s|\.dir/|: |; s|\.o$||'
    echo
    echo "Top 30 dead headers:"
    head -30 "$out/dead-headers.txt"
    echo
    echo "Top 30 dead symbols (demangled, noise-filtered):"
    head -30 "$out/dead-symbols.txt"
    echo
    echo "Top 20 compile warnings:"
    head -20 "$out/compile-warnings.txt"
} | tee "$out/summary.txt"
