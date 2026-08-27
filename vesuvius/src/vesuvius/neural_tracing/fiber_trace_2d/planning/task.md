# Task: Interleaved-lattice winding inference

Replace independent integer winding inference with a joint orientation and
winding model for crop `direction-ablation --bp-only` processing.

- Every split piece remains an independent variable.
- The two physical fiber orientations occupy separate integer lattices:
  class A has coordinate `k`, class B has coordinate `k + phase`, with integer
  `k` and one shared fractional `phase`.
- Physical H/V naming is unobservable. Keep the existing crop-central class-A
  seed as a deterministic gauge; a global class swap is equivalent.
- Retain an explicit Mixed/error state. It must disable orientation propagation
  rather than encourage neighboring pieces to become Mixed.
- Fit a positive global winding-measurement scale so systematically
  under-calibrated integrals, such as `0.8` per full winding, can map to one
  latent winding unit.
- Solve the discrete piece states and global continuous phase/scale by
  deterministic alternating inference. Use soft BP beliefs when refitting the
  global parameters and deterministic multiple initializations.
- Preserve short output names and publish consecutive nonnegative winding
  indices, with solver-relative values and fitted calibration in CSV reports.
- Validate on synthetic two-lattice chains and the existing 384-base crop.

## Progress-reporting follow-up

Add visible progress output for the interleaved winding solve. The output must
identify the multi-start initialization, calibration round, adaptive-support
round, and inner BP message iteration, without changing numerical behavior.

## Input-quality filtering follow-up

Add a CLI flag that retains only the best-quality stored crop traces before
direction classification, splitting, constraint extraction, BP, and
visualization. Reuse the established cost-density ordering.
