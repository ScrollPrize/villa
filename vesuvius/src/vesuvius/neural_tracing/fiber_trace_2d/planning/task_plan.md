# Plan: eight-GPU S1a retraining configuration

## Active follow-up: sample-count logging

Use a sample-axis TensorBoard writer adapter for scalar and image events so evaluation, timing and visualization share the training-sample coordinate. Keep text metadata at step zero. Console metrics show samples and explicit update count. Keep internal step, checkpoint filenames, scheduling, sampling and profiling JSON in optimizer-update units. Store samples_seen and effective global batch in checkpoints; resume uses saved count plus new updates times new global batch. Old checkpoints infer only from a recorded global batch, or require training.resume_samples_seen if historical world size is unknown. Test real TensorBoard events, checkpoint roundtrip and changed-batch resume, legacy handling and timing output. Document historical event axes remain unchanged; no automatic rewrite of old runs. Obtain independent plan and implementation review.

Scope correction: only create the fast reproduction, label s1a_repro1. Compare against existing runs. The earlier matched-control proposal below is superseded; its config and commands were removed at the user's request.

Use saved training_baseline_s1a_20260729.json datasets, transforms, augmentation, evaluation and seed. Create a fast config using the measured GroupNorm/output-stride-4/BF16/global-batch-128 setup, LR 0.01, 100 warmup updates and no clipping. Enable measured shared sample pipeline, async preparation and all CUDA value augmentation. Create a matched standard-loader control with the same model, data, batch and optimizer to isolate the pipeline change. Distinguish this from the historical BatchNorm/stride-1/global-batch-64 baseline. Run length 20,000 updates with evaluation/checkpoints/TensorBoard retained. Prior LR trial predates global loss normalization: recommendation, not guaranteed optimum.

## spec update

Default console/TensorBoard progress now uses cumulative global training samples. Document checkpoint sample metadata, changed-batch resume and explicit legacy override. Internal scheduling continues to use updates.

## docs updates

Add eight-GPU commands, comparison caveats, output paths and validation record to a retraining guide and task log. Add changelog entry.

Follow-up: launch from /ephemeral/hendrik/fiber3d_train2 with absolute config paths, write runs beneath its runs directory, and use short first-reproduction labels s1a_repro1_fast/std. Validate config loading from that working directory.

## validation

Independent plan review. Parse configs with actual loader/model APIs; compare settings except explicit pipeline options and run names; verify source globs. Run relevant existing config/model/training-option tests without starting training. Verify imports resolve to villa2 and report dependency limitations. No installs or training launch.
