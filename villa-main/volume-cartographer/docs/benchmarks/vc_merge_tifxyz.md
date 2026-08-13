# `vc_merge_tifxyz` benchmarks

This file tracks wall-clock and peak resident-set-size numbers for
`vc_merge_tifxyz` runs. Add a new row whenever a PR may shift its cost.

## Methodology

Build with **`Release`** for headline numbers (`RelWithDebInfo` is
acceptable for relative comparisons across PRs as long as the row
records the build type — same `-O3` codegen with debug info added).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target vc_merge_tifxyz vc_obj2tifxyz_legacy
```

Run, on the dataset under measurement:

```sh
for i in 1 2 3; do
  /usr/bin/time -v build/bin/vc_merge_tifxyz --merge <volpkg>/<run>.merge.json
done 2>&1 | tee docs/benchmarks/raw-$(date +%Y%m%d).txt
```

`/usr/bin/time` flags vary by platform: GNU `time` uses `-v`, BSD/macOS
`time` uses `-l`. Both report wall-clock and peak resident-set size;
only the format of the output differs.

Report the **median** of the three runs for both wall-clock and peak
RSS. Record CPU model and arch (e.g. via `lscpu` / `uname -m` on Linux,
`sysctl -n machdep.cpu.brand_string` on macOS).

## GUI

The Qt feature wired into VC3D launches `vc_merge_tifxyz` as a
subprocess via `CommandLineToolRunner`; it adds dialog construction,
JSON pretty-printing, and `QProcess` startup but no hot loop. There is
**no separate GUI benchmark** — latency from button click to first
stdout line is dominated by `vc_merge_tifxyz` startup (covered above).
Re-litigate this only if a profiler shows otherwise.

## Results

| Date       | Git SHA  | Build type | CPU       | Arch    | Dataset       | Median wall-clock | Median peak RSS | Notes |
|------------|----------|------------|-----------|---------|---------------|-------------------|-----------------|-------|
| _add row_  | _short_  | _Release_  | _model_   | _e.g._  | _path / size_ | _e.g. 8.4s_       | _e.g. 612 MiB_  | _N=3, default flags_ |

When the `s1_ds2.volpkg` minimal pull (see
`scripts/download_minimal_volpkg.sh`) is available locally, the
canonical baseline is a 1×2 horizontal merge of the two tifxyz dirs
the script downloads. Document in the row which two surfaces were
used.
