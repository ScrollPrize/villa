# Plan: separate prepass and winding Defect controls

## CLI and solver configuration

1. Keep `--bp-mixed-cost` scoped to the ordinary H/V/Mixed orientation BP.
2. Add an independent finite nonnegative `--winding-defect-cost` option with
   default `0.5`. In fixed-prepass mode it is the late-Defect unary in both
   backends. In non-fixed joint-grid mode it is the joint solver's only Defect
   unary. Non-fixed alternating continues to use the prepass posterior as its
   orientation prior and does not charge this unary again.
3. Accept the winding option only with
   `--bp-only --bp-inference sum-product-mixed`, matching the other winding
   controls. Do not make either cost silently inherit the other.
4. Preserve the initial prepass cost in the ordinary BP report. Add an
   explicit winding Defect cost to the interleaved winding report and its
   console/CSV diagnostics so joint-only output is not mislabeled as a
   prepass cost.

## Prepass artifacts

1. When the winding report declares `FixedPrepass`, translate its exact fixed
   orientation vector to the shared ternary state representation.
2. Reuse `writeFiberletCropTernaryStateObjs` to write
   `<base>_prepass_v.obj`, `<base>_prepass_err.obj`,
   `<base>_prepass_h.obj`, and `<base>_prepass_tie.obj`.
3. Keep final `<base>_{v,err,h,tie}.obj` output unchanged. The prepass tie file
   is expected to be empty because exact prepass ties are fixed as Defect, but
   retaining the common four-file contract avoids a second exporter.

## Testing

1. Add focused solver coverage for both fixed-prepass backends proving the
   winding report records its separate Defect cost while the exact fixed
   orientation vector remains unchanged. Verify non-fixed alternating retains
   its posterior-prior behavior.
2. Verify exact fixed H/V/Defect assignments map to the expected prepass OBJ
   layers and no prepass artifacts are claimed for joint orientation mode.
3. Exercise CLI parsing for independent defaults, distinct explicit values,
   invalid negative/non-finite values, and the required Mixed BP mode.
4. Build the CLI and focused winding/crop tests with 32 jobs; run focused CTest
   and `git diff --check`.

## Spec update

Specify that `--bp-mixed-cost` belongs to the orientation prepass,
`--winding-defect-cost` belongs to winding inference, their defaults are
independent, and fixed-prepass runs persist the exact consumed assignment as
separate OBJ layers.

## Docs updates

Update `volume-cartographer/docs/fiber_chunk_tracing.md` with both unary costs,
their scopes, a fixed-orientation invocation example, and the prepass artifact
names.

## Changelog

Record the independently tunable winding Defect unary and inspectable prepass
OBJ partition.
