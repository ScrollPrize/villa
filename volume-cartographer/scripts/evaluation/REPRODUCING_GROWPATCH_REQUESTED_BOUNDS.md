# Check requested resume growth bounds on real input

Use an already built current official `vc_grow_seg_from_seed` CLI and existing Python >=3.9 with NumPy/Pillow. The opt-in checker downloads or installs nothing. Supply a small fully valid rectangular real-scroll tifxyz resume mesh with nonzero saved generations, the matching local prediction volume, and its numeric/loss configuration:

```text
python check_growpatch_requested_bounds.py --binary /path/vc_grow_seg_from_seed --volume /real/prediction.zarr --resume /real/mesh --params /real/params.json --output /new/path/results --timeout 600
```

The PR description pins the accompanying evidence commit, which provides `fetch_real_fixture.py` and its public S3 source manifest. After fetching those real inputs into a new directory, use `--fixture-root /new/path/fixture` instead of the three input arguments above, retaining explicit `--timeout 600`. This external watchdog override leaves the executed checker's default 120 and numerical settings unchanged.

The checker runs zero extra rows/columns, one allowed right column, and a wider positive control. It preserves the numeric loss configuration while forcing regular resume growth, `growth_scale=0`, no inpaint/rewind, one thread, fixed seed 2104, and small generation counts. Each run saves exact parameters, argv, binary hash, process status and complete logs. Output directories must be new and separate from inputs; input files are hash-checked before/after.

The observer reads actual XYZ/generations TIFFs and metadata grid offsets. Zero expansion must preserve original coordinates exactly; the right-column and wider controls must make permitted progress. Exit 0 means all three completed cases passed; exit 1 is a completed requested-grid violation; exit 2 is an input/native/timeout/output failure. A successful native exit alone is insufficient.

This tests grid admission on the supplied data. It does not measure physical sheet or ink quality. The small public fixture uses real, reversibly translated XYZ with a newly initialized generation-one test channel; it does not claim to retain a public original tracing history. Fresh-seed, inpaint and coarse-grid runtime coverage must be reported separately.
