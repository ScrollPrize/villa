# Reverse-consistency winding BP: message-limit sweep

Controlled follow-up on the same2048crop and reverse-v2 traces. No new
inference, tracing, reference conditioning or constraint removal. References
are read independently against frozen initial BP, with existing GT global
sign/offset alignment for comparison.

## Isolation and validation

The original `--bp-message-iterations` also controls normal-lattice alignment
and the fixed-orientation prepass. An initial shared-cap1000pilot was stopped
and is excluded. Added optional `--winding-message-iterations` so these
preprocessing passes remain at500. Default behavior is unchanged.

Build: approved `bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build`.
Tests:29passed using
`/home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_independent_reference_cli.py vesuvius/tests/neural_tracing/test_crop_reverse_cli.py`.

The rebuilt500control exactly reproduces the historical reference candidate
table and factor CSV. At1000the candidate table, factor CSV, preprocessing
classes and all6790decoded piece validity/winding/final-class assignments are
unchanged from500.

## Results

|Winding update cap|Correct references|No hard-sign errors|Final residual|Converged|Total wall time|
|---:|---:|---:|---:|---|---:|
|500 control|38.46%(10/26)|34.62%(9/26)|532.897487|No|111.59s|
|1000|38.46%(10/26)|34.62%(9/26)|532.549403|No|193.25s|
|5000|38.46%(10/26)|34.62%(9/26)|532.329718|No|777.39s|

All runs completed successfully but reached their message cap. Pure winding
solve times were60.547383s,140.736602s,727.630210s respectively.
All26top-ranked reference rows (including finite costs and hard-sign errors)
are unchanged; the full ladder is in
[the baseline report](2026-09-23-reverse-consistency-independent.md).
Each setting has271summed hard-sign violations in the independent reference
readout; these are not counts of unique violations in the solved BP graph.
At5000two of6790decoded piece validity/winding/final-class records change;
the full top5candidate table is therefore not byte-identical, unlike1000.
This does not affect any top-ranked reference result.

All three runs have identical executable/shared-library/input/reference hashes,
byte-identical factor CSVs, and identical orientation prepass classes. The500
control reproduces the historical result despite rebuilding the CLI.

Conclusion: simply allowing2x or10xmore winding updates does not fix the short
ladder. Messages remain unconverged and fluctuate, so this does not establish
that a genuinely converged or different solver would give the same result.
Further iteration-budget increases alone are unsupported by these results;
damping/scheduling or order-graph diagnostics would be separate experiments.

RelWithDebInfo,128workers, one run per setting. Times include preprocessing
and reference readout; they are not repeated throughput estimates.

## Reproduce

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_independent_reverse_message_limits_20260923.json
```

Outputs are under `/ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/`,
prefix `bp-beam-new2048-coverage40x80-angle45-reverse-v2-independent-mN-fixedprep`,
where N is500,1000or5000. Each completed run includes `.run.json` hashes and
command, `.log`, `_consistency.csv` and `_winding_factors.csv`.
