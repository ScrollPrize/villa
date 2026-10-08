# Combined JSON/NML finetuning at LR0.001

The previous combined run was stopped after complete raw and EMA snapshots at
update21100 (2700800 samples). Resume uses that run's `snapshots/current.pt`,
not a best or EMA export. Optimizer state and both EMA tracks are retained.

Config: `fiber_trace_3d/configs/train_finetune_5scroll_s1a_ema_lr0001.json`.
Only changes from the exact source launch config are LR0.01→0.001, run name,
and raw resume path. All datasets, sampling weights, augmentations, testing,
EMA decays and the2Mupdate limit remain unchanged. Restored optimizer groups
receive the configured LR; the completed100update warmup is not repeated.

Started using the approved `vesuvius/scripts/run_fiber3d_experiment
production-start` command after its bounded stop watcher confirmed the old root
had exited. Manager support remains automatic. New config calibration uses real
training updates under an exclusive grant before registering adaptive training.

Launch log:
`/ephemeral/hendrik/fiber3d_train2/runs/.launches/finetune_5scroll_s1a_128_sd1_16x8_ema_lr0001-20260912_223739/train.log`.

CPU launcher regression suite:22passed. Exact config diff independently reviewed.
Startup verified checkpoint_step21100, next_step21101, optimizer_lr=[0.001],
and manager phase running under the exclusive calibration grant. Run directory:
`/ephemeral/hendrik/fiber3d_train2/runs/finetune_5scroll_s1a_128_sd1_16x8_ema_lr0001_20260912_223743`.
