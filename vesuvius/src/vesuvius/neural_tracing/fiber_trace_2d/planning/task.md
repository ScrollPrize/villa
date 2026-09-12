# Long S1a training with historical observation and EMA

Reproduction is complete (cffe3cde9). Keep modern training/data/augmentation and
current test metrics. Start fresh S1a NML training in the regular
/ephemeral/hendrik/fiber3d_train2/runs directory for2000000 optimizer updates.
Add historical average angle and total loss as additional observation-only test
metrics, not replacements or checkpoint-selection inputs. Add inexpensive EMA
tracks, separately evaluated and saved so their quality can be compared later.
Preserve manager-controlled automatic calibration/pause/resize and resumability.
Measure overhead before launching; all GPU tests use manager allocations and
the already approved configurable experiment/test launchers.
