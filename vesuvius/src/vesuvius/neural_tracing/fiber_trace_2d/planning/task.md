# Fiber training reproduction

Reproduce old S1a NML performance: old-test mean direction error below about20degrees at200k samples. Each trial trains300k samples; compare200k–300k to s1a_128_1_single_8x8_20260729_075013. Recover test data/sampling from s1a_128_1_single_8x8_20260730_063937.

Stack: historical test sampling; SyncBN; stride4 upsample trilinear/nearest + old full-resolution loss; full-resolution decoder with dense/points-only NML; BS64 LR.01 and BS128 LR.02; fixed on/off blur sigma; old mask handling. Once successful ablate changes individually; otherwise exhaust these and sensible evidence-driven options. Fresh weights. Preserve stage commits/branches and brief results in task_log.md.

Use approved run_fiber3d_experiment and git prefixes, venv_las, artifacts under /ephemeral/hendrik/fiber3d_train2. Every GPU use requires manager exclusive nonpausable/nonadaptive grant or full adaptive support. Continue autonomously until culprit found or options exhausted.
