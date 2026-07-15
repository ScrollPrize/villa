# 3D Default VC3D Cache Budget

Set the 3D fiber Python-side default VC3D decoded/hot chunk cache budget to
512 MiB per loader/worker.

Required behavior:

- If `volume_cache_memory_mib` is missing or `null` in a 3D config, Python must
  pass a 512 MiB cache budget to VC3D instead of falling through to VC3D's 8 GiB
  default.
- If a positive `volume_cache_memory_mib` is provided explicitly, keep honoring
  that value.
- Apply the same default to inline/config-from-mapping 3D configs and the
  generated 2D Trace2CP geometry loader used by 3D evaluation.
- Do not change batch size or loader worker counts.
- Update tests/docs/spec/task log.
