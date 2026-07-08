new task - we will now start work on a learned 2d fiber refinement / method.
the current plan is shown in vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/plan.md
please write task_plan.md in the same dir that:
- data loading: copy fiber parsing from the existing neural_tracer/fiber code vesuvius/src/vesuvius/neural_tracing/fiber_trace/
- writes an initial data-loader/iterator & augmentation code
- there should be fiber side-strip code in vc3d - possibly also a version in lasagna or the existing fiber-trace code
- scope for now - just implmement initial batch loading of fiber-strip patches around random cps from the fiber dataset
- load several (for now +- 7/8 vx - so 16 overall) patches around the cp strip (with the +- offset being along the strip z dir)
- write the dataloader and a small data-loader-tester/runner that will load a batch from a specified cp (where specified cp is a deterministc random idx into the fiber dataset)
- each patch should be sampled independently
- we should link/use vc3d sampling methods - specifically we want to sample the patch as a surface/segment - so we can modify coords before loading for the later augmetations (so we do initially want to load just a plane, but we'll use the more generic coords functions anyways so we can get augmentable patches later
- use a prefetch approach - where we check which chunks we need first and download them using the neural-tracer/vesuvius chunk fetching into the cache
- the dataset and training config should be a vesuvius style json
- parts you'll need to inspect (potentially):
    - vc3d fibers & jsons & fiber-side-strip creation
    - vesuvius for chunk fetching (potentially neural tracer?) - prefer plain vesuvus over neural tracer dependenceis where possible
- write docs into docs/ that explain the code structure
- also update specs.md with bullet points containint all the implemented specs from both plan.md and task.md
NOTE your task right now is to:
- write task_plan.md for later implenentation
- update specs.md
nothing else, the rest above is just what should GO into those two!
