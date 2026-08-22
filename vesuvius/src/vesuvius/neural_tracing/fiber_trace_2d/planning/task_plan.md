# Plan: fixed nonlinear uint16 fiberlet costs

1. Add an explicit fixed-sqrt cost domain and fixed density ceiling to replay
   scenarios. Keep raw-total affine encoding bitwise unchanged.
2. Encode
   `round(65535 * sqrt(clamp((total / path_length) / 256, 0, 1)))`, then decode
   `256 * (code / 65535)^2 * path_length`. Reject invalid costs, lengths, and
   ceilings.
3. Apply the identical scalar transform in eager and cache-backed evaluation.
   Do not scan owner chunks or include cost-domain settings in geometry cache
   identity, fingerprints, or persisted payloads.
4. Add `compact_axis_cost_sqrt_u16_max256`; record the domain and ceiling in
   replay JSON and machine-readable output.
5. Add focused codec, saturation, fixed-range, cache-profile, and scenario
   matrix tests. Build with `-j32` and run the fiberlet storage, replay, and
   paths tests, reporting pre-existing failures separately.
6. Run the full-fiber Paris4 radius-768 scenario with the existing hot compact
   cache at beam 16, `H=384`, `D=48`, and exact search. Verify the cache file
   count and fingerprint are unchanged.
7. Update the specification, user documentation, changelog, status, and task
   log with the exact mapping and measured result.
