# Merge Fiber 3D Extension And Adapt Multi-Dir Config

Merge the current `fiber-3d-ext` branch into the active multi-direction 3D
fiber training branch, then adapt the newly added actual 64-scale S1A NML
training config to the multi-direction model output.

Requirements:

- Preserve the `fiber-3d-ext` requested-level blocking coordinate sampling and
  native 3D Trace2CP rendering changes.
- Preserve the current multi-direction 3D fiber direction/presence training
  implementation.
- Adapt the newly added
  `fiber_trace_3d/configs/train_s1a_nml_all_64_sd2.json` config to use two
  direction/presence branches.
- Resolve merge conflicts without pulling unrelated old stash work into the
  branch.
- Run focused validation for the touched 3D Python training path and the merged
  Trace2CP/native tests.
