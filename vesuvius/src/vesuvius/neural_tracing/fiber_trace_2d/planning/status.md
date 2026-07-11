# Loader Serialization Bottleneck Status

- [x] Read task/spec/plan context.
- [x] Establish current load-only benchmark baseline.
- [x] Inspect Python loader and VC3D sampling/cache boundaries.
- [x] Rule out volume-sampling finish sharding as the primary fix.
- [x] Rule out isolated whole-batch loaders as the primary fix.
- [ ] Move per-sample CUDA coordinate materialization out of loader workers.
- [ ] Batch geometric coordinate materialization across the loaded batch.
- [ ] Update specs/docs/log.
- [ ] Run focused tests.
- [ ] Re-run benchmark and record before/after numbers.
