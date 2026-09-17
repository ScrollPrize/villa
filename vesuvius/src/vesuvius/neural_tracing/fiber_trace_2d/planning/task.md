# Frozen refined-fiber split-range embedding training

Start a fresh embedding-model training run on the latest fibers produced by the
stopped full-resolution refinement run. Online refinement must be disabled: the
imported refined training geometry remains fixed throughout training. Evaluation
uses immutable original test GT; it does not refine or invent paired test state.

Keep 16 embedding output channels, split into two independently supervised banks:

- channels 0:8 are short-range embeddings supervised within each tile;
- channels 8:16 are long-range embeddings supervised between paired patches from
  the same fiber.

Preserve the existing paired sampling/value-augmentation contract, presence-based
negative importance sampling, five-scroll JSON dataset, LR0.01, global batch128,
managed adaptive GPU lifecycle, EMA and full-resolution model output. Initialize
the model, optimizer, EMA and update/sample counters from scratch; import only the
refined fiber archive from the stopped run.
