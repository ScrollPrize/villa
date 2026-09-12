# Plan

1. Copy current five-scroll config settings into train_s1a_nml_16x8.json. Use
   existing S1a NML dataset and affine from train_s1a_nml_8gpu_fast.json; retain
   its held-out JSON test sources without the NML transform. Enable PHercParis4
   scroll ID and dense supervision for both, retaining current arc-length sampling.
2. Keep current model/augmentation, LR0.01, B16/global128, 2M steps; user follow-up sets test/checkpoint intervals100,
   no embeddings and manager handling. Give it a distinct s1a_nml_128_sd1_16x8 run name.
3. CPU config-contract tests compare all other fields against collection config,
   verify exact source/transform and confirm local NML/test file availability.
   Use approved fixed test launcher; no CUDA test needed for config-only change
   with already-tested unchanged model/target code.

## Spec update

No new semantics; current dense balanced sampling specs apply to a single scroll.

## Docs updates

Document config, test sources, sampling and launch command in collection docs;
update task log/status. This is reproduction of the old dataset, not old numeric
training trajectory or CP-only test metric. No automatic commit requested.
