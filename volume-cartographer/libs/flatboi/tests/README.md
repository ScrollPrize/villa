# SLIM assembly regression checks

The regression executable compares direct triangle assembly with the generic
sparse-product implementation. It covers supported energies, soft constraints,
cache ownership, reinitialization with changed topology, tetrahedral fallback,
input bounds and nonfinite-result rejection. Synthetic cases do not require
downloads. An optional OBJ adds real-surface coverage.

## Build and run

Install the native dependencies described in the [VC3D build instructions](../../../README.md).
From the repository root, use your configured build directory:

```sh
cmake -S volume-cartographer -B build \
  -DVC_BUILD_FLATBOI=ON -DFLATBOI_BUILD_SLIM_TESTS=ON
cmake --build build --target check_slim_assembly --parallel 1
```

`check_slim_assembly` builds the executable and runs it with two PaStiX/BLAS/OpenMP
threads. A failed assertion or invalid numerical comparison exits nonzero.
The Linux flatboi CI job runs this target alongside the application build.
Named cases use the repository's `vc_test.hpp` runner. Numerical failures
report the energy, iteration and matrix coefficient or RHS row.

With `VC_TESTING=ON`, the same executable is registered in CTest under the
`vc-flatboi` label. `FLATBOI_BUILD_SLIM_TESTS` defaults to `VC_TESTING` in a fresh
configuration; an explicit cached value takes precedence.

```sh
cmake -S volume-cartographer -B build \
  -DVC_BUILD_FLATBOI=ON -DVC_TESTING=ON -DFLATBOI_BUILD_SLIM_TESTS=ON
cmake --build build --target test_slim_assembly --parallel 1
ctest --test-dir build --output-on-failure -R '^test_slim_assembly$'
```

## Real-surface comparison

Supply an OBJ with positions, triangular faces and a consistent UV mapping for
every vertex, such as the output of `vc_tifxyz2obj`. Use a result path that does
not already exist:

```sh
PASTIX_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  build/bin/test_slim_assembly surface.obj result.json
```

Missing UVs, invalid indices and conflicting UVs at a shared vertex are
rejected with an error before assembly. Separate texture indices may share
one vertex when their UV coordinates agree.

The JSON contains maximum scaled matrix/RHS error, absolute lifecycle error,
comparison counts and mesh dimensions. The matrix/RHS acceptance condition is
`abs(actual - reference) <= 1e-10 + 1e-10 * abs(reference)`. Nonfinite coefficients,
RHS values or derived errors fail before reductions; the test exercises NaN and
both infinity signs in each comparison operand. Lifecycle solutions must differ
by less than `1e-9` in maximum absolute value.
