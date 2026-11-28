"""Mutation check for sparse_pool_metal: each mutant of the kernels must fail test_sparse_pool_metal.py.

    python tests/sparse_pool_metal_mutants.py [--tolerance-only] [-k NAME]

Each mutant is a textual edit of sparse_pool_metal.py (every edit must match the expected number of
times, so that no mutant is silently a no-op), written to a temporary directory that comes first
on sys.path; the tests find sparse_pool_ops and the cases in the real tree. The compile-time
self-check is disabled in the mutants, so the tests alone must catch each one, and the tests of
the self-check itself are deselected. --tolerance-only also deselects the bitwise test, for
machines where the kernels are built with contraction on and only the tolerance tests apply.

The unmutated copy must pass, the canary (a syntax error) must fail, and the mutants listed as
equivalent are expected to survive: they cannot change any result (reasons below).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

LASAGNA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(LASAGNA, "tests", "test_sparse_pool_metal.py")

LOCAL = ["const float {a} = p{a2} - c{a2}f * 32.0f + 1.0f;".format(a=a, a2=a[1]) for a in ("lx", "ly", "lz")]

MUTANTS = {
    # the usual slips: halo, rounding, neighbourhood, gradient, scaling, slot stride, masking
    "no_halo": [(s, s.replace(" + 1.0f;", ";"), 1) for s in LOCAL],
    "round_half_even": [("int(round(val))", "int(rint(val))", 1)],
    "six_neighbourhood": [("                if (table[long(z) * ip[6] + long(y) * ip[7] + long(x) * ip[8]] >= 0) {",
                           "                if (abs(z - cz) + abs(y - cy) + abs(x - cx) > 1) {\n"
                           "                    continue;\n                }\n"
                           "                if (table[long(z) * ip[6] + long(y) * ip[7] + long(x) * ip[8]] >= 0) {", 1)],
    "swap_dfx_dfy": [("t = go * dfx * isx; dgx = dgx + t;", "t = go * dfy * isx; dgx = dgx + t;", 1),
                     ("t = go * dfy * isy; dgy = dgy + t;", "t = go * dfx * isy; dgy = dgy + t;", 1)],
    "drop_inv_scale": [("t = go * dfx * isx;", "t = go * dfx;", 1), ("t = go * dfy * isy;", "t = go * dfy;", 1),
                       ("t = go * dfz * isz;", "t = go * dfz;", 1)],
    "slot_stride_ignores_C": [("c.base = slot * ip[0] * BRICK", "c.base = slot * BRICK", 1)],
    "unloaded_not_masked": [("const long slot = long(table[long(czf) * ip[7] + long(cyf) * ip[8] + long(cxf) * ip[9]]);",
                             "const long slot = max(long(table[long(czf) * ip[7] + long(cyf) * ip[8] + long(cxf) * ip[9]]), 0L);", 1)],
    # more
    "no_clamp_at_33": [("ix1 = clamp(ix0 + 1, 0, MAX_LOCAL);", "ix1 = ix0 + 1;", 2)],
    "trunc_chunk_index": [(f"floor(p{a} / 32.0f)", f"trunc(p{a} / 32.0f)", 1) for a in "xyz"],
    "bounds_off_by_one": [("cxf < float(ip[3])", "cxf <= float(ip[3])", 1)],
    "channel_offset_ignored": [("pool + c.base + channel * BRICK", "pool + c.base", 1)],
    "u8_truncates": [("int(round(val))", "int(val)", 1)],
    "f32_channel_major_index": [("out[ch * N + n] = interpolate(c, corners(pool, c, ch));",
                                 "out[long(n) * C + ch] = interpolate(c, corners(pool, c, ch));", 1)],
    "grad_drops_grad_output": [("t = go * dfx * isx;", "t = dfx * isx;", 1)],
    "grad_wrong_difference": [("float dfy = ofx * ofz * (v.v010 - v.v000);", "float dfy = ofx * ofz * (v.v010 - v.v001);", 1)],
    "missing_marks_loaded_chunks": [("continue;  // loaded", ";  // loaded (mutant: not skipped)", 1)],
    "missing_rounds_not_truncates": [("return min(int(q), n - 1);", "return min(int(rint(q)), n - 1);", 1)],
    "missing_scales_by_reciprocal": [("precise::divide(precise::divide(v - origin, spacing), 32.0f)",
                                      "precise::divide((v - origin) * (1.0f / spacing), 32.0f)", 1)],
    "missing_bit_order_reversed": [("const uint cell = uint(w) * 32u + ctz(b);\n            b &= b - 1u;",
                                    "const uint k = 31u - clz(b);\n            const uint cell = uint(w) * 32u + k;\n"
                                    "            b &= ~(1u << k);", 1)],
    # (scan mutants that keep every write inside the output: an earlier start, a skipped word)
    "missing_block_starts_early": [("    long pos = starts[t];", "    long pos = (t == 0) ? 0 : starts[t - 1];", 1)],
    "missing_range_drops_last_word": [("w1 = long((end - 1u) >> 5) + 1;", "w1 = long((end - 1u) >> 5);", 1)],
    "missing_range_starts_a_word_late": [("w0 = long((~bits[words]) >> 5);", "w0 = long((~bits[words]) >> 5) + 1;", 1)],
    "missing_range_from_one_lane": [("const uint group_first = simd_min(first);", "const uint group_first = first;", 1),
                                    ("const uint group_end = simd_max(end);", "const uint group_end = end;", 1)],
    "missing_count_from_last_block_start": [("staged[0] = int(starts[ip[1]]);", "staged[0] = int(starts[ip[1] - 1]);", 1)],
    # positive control: the mutated copy is the one imported
    "canary_syntax_error": [("kernel void sparse_sample_f32(", "kernel void sparse_sample_f32(+", 1)],
}

# Mutants that cannot change a result, and why. They are run and must survive.
EQUIVALENT = {
    # A non-finite p fails the float bounds check anyway (comparisons with NaN are false, and +-Inf
    # is outside [0, cX)), as long as the library keeps safe math (compile() checks).
    "isfinite_dropped": [("    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) {\n        return false;\n    }\n", "", 1)],
    # l >= 1, so floor(l) >= 1 and clamping i0 before or after i1 = i0 + 1 gives the same corners.
    "u8_clamps_first": [("if (!locate<false>(", "if (!locate<true>(", 1)],
    # The interpolated value is in [0, 255 (1 + eps)], so the clamp never changes it.
    "u8_no_clamp": [("uchar(clamp(int(round(val)), 0, 255))", "uchar(int(round(val)))", 1)],
    # Dividing by 32 is exact in float32 (normal range): (x / s) / 32 == x / (32 s), both rounded once.
    "missing_divides_once": [("precise::divide(precise::divide(v - origin, spacing), 32.0f)",
                              "precise::divide(v - origin, spacing * 32.0f)", 1)],
}

SELF_CHECK = ("        _self_check()\n", "        pass  # self-check disabled for the mutation check\n", 1)


def _mutated_source(edits):
    with open(os.path.join(LASAGNA, "sparse_pool_metal.py")) as f:
        source = f.read()
    for old, new, count in [SELF_CHECK, *edits]:
        found = source.count(old)
        if found != count:
            raise SystemExit(f"edit matches {found} times, expected {count}: {old[:80]!r}")
        source = source.replace(old, new)
    return source


def _run(name, edits, deselect):
    tmp = tempfile.mkdtemp(prefix=f"metal_mutant_{name}_")
    try:
        with open(os.path.join(tmp, "sparse_pool_metal.py"), "w") as f:
            f.write(_mutated_source(edits))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([tmp, LASAGNA]), PYTHONDONTWRITEBYTECODE="1")
        env.pop("LASAGNA_METAL", None)
        # cwd = tmp: `python -m pytest` puts the working directory first on sys.path.
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", TESTS,
                              "-k", deselect], cwd=tmp, env=env, capture_output=True, text=True)
        lines = [x for x in out.stdout.splitlines() if x.startswith("FAILED") or " passed" in x or " failed" in x
                 or "error" in x.lower()]
        return out.returncode, (lines[-1] if lines else out.stdout[-300:]).strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance-only", action="store_true", help="also deselect the bitwise test")
    ap.add_argument("-k", dest="only", help="run only the mutants whose name contains this")
    args = ap.parse_args()
    deselect = "not self_check" + (" and not bitwise" if args.tolerance_only else "")
    rc, line = _run("unmutated", [], deselect)
    print(f"unmutated: {'PASS' if rc == 0 else 'FAIL'}  {line}", flush=True)
    if rc != 0:
        raise SystemExit("the unmutated copy must pass")
    caught = survived = 0
    problems = []
    for group, expect_caught in ((MUTANTS, True), (EQUIVALENT, False)):
        for name, edits in group.items():
            if args.only and args.only not in name:
                continue
            rc, line = _run(name, edits, deselect)
            is_caught = rc != 0
            caught += is_caught
            survived += not is_caught
            status = "caught" if is_caught else "SURVIVED"
            if is_caught != expect_caught:
                problems.append(name)
            label = "" if expect_caught else " (equivalent: expected to survive)"
            print(f"{name}: {status}{label}  {line}", flush=True)
    print(f"\n{caught} caught, {survived} survived; unexpected: {problems or 'none'} ({deselect})")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
