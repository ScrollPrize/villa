# Local Development

The current checkout has a generic Release test tree at `build-release`.

Use all 32 local build cores:

```bash
cmake --build volume-cartographer/build-release --parallel 32 --target <target>
```

The synthetic rendering benchmark requires a generic non-native Release build,
Valgrind on Linux amd64, and `VC_RUN_RENDER_BENCHMARKS=ON`. See
`docs/benchmarks/synthetic_rendering.md` for full configure and run commands.
