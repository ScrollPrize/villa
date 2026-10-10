# Evidence summary

## Evidence identity

- Contribution: metadata-only frame provenance preflight
- Pull request: <https://github.com/ScrollPrize/villa/pull/1944>
- Validated commit: `917de185c102c045af1b6523ea11e68736a905e2`
- Catalog snapshot SHA-256: `40517076dc5f2128599868b337e25a17922ebe70a22b238b990e109b6f82af0f`
- Catalog snapshot size: 17,537,378 bytes

## Technical validation

| Check | Result |
|---|---:|
| Frame-preflight tests | 20/20 passed |
| Surface-preflight regression tests | 30/30 passed |
| Total | 50 passed, 0 failed, 0 skipped |
| Python | 3.14.8 |
| pytest | 9.1.1 |
| `compileall` | PASS |
| `git diff --check` | PASS |

All P1, N1, and U1 cases were executed twice. Each pair was byte-identical, and the outputs were also byte-identical to the corresponding locally archived LIMIT-003 outputs.

## Real-data catalog cases

| Case | Frame relation | Frame reason | Transform direction | Alternative relation | Original SHA-256 | Sanitized derivative |
|---|---|---|---|---|---|---|
| P1 | `TRANSFORM_REQUIRED` | `EXPLICIT_DIFFERENT_TARGET_AND_DIRECTED_VOLUME_TRANSFORM` | `FORWARD` | `SAME_FRAME_ALTERNATIVE_AVAILABLE` | `f61f0c17fa4970b233260adc26949b0e622cb37acf3a395fbb22283c913b102b` | `examples/p1_example.json` |
| N1 | `SAME_FRAME` | `EXPLICIT_ENTRY_TARGET_MATCHES_REQUESTED_VOLUME` | — | `NONE_CONFIRMED` | `2feba879b9df1cd258abb5f8cb03ec845f1312b8c7268814eba61bd491f42cbc` | `examples/n1_example.json` |
| U1 | `UNKNOWN` | `MISSING_EFFECTIVE_ENTRY_FRAME_ID` | — | `SAME_FRAME_ALTERNATIVE_AVAILABLE` | `80fdd5d47357a69fcb210e88d5fb0fc1fdf2538eb77ab91dd24d501c1e060845` | `examples/u1_example.json` |

The public examples are sanitized derivatives. Their own hashes are recorded in `provenance/public_manifest.json` and must not be confused with the original hashes above.

## Resource measurement

The representative N1 CLI run was measured on Windows using `System.Diagnostics.Process`:

- `WorkingSet64` sampled every 5 ms;
- `PeakWorkingSet64` queried after process exit;
- reported peak is the maximum of both measurements;
- peak: 90,927,104 bytes, or 86.715 MiB;
- process exit code: 0.

This single measurement is not a general upper bound for all catalogs, platforms, Python builds, or cases.

## Known limits

- No independent external reproduction has been completed.
- No community use has been demonstrated.
- There is no automatic renderer integration.
- The evidence does not validate CT voxels, TIFXYZ coordinate rasters, or Zarr chunks.
- The evidence comprises three deliberately selected real-data catalog cases.
- The catalog snapshot is identified by size and hash but is not bundled.
- Metadata provenance does not establish mesh quality, transform accuracy, or render quality.
