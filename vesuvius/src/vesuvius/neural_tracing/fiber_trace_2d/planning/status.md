# Native Fiber Trace Locality And Scheduling Optimization Status

- [x] Read repository and fiber-trace workflow instructions
- [x] Replace the prior active task, plan, status, and log
- [x] Capture the retained 1.869s / 7-restart baseline
- [x] Write implementation, testing, benchmark, spec, and docs plan
- [x] Review plan directly against current specs and architecture
- [ ] Add result-neutral locality and scheduling instrumentation
- [ ] Request approval and benchmark instrumentation
- [ ] Test worker granularity
- [ ] Test deterministic top-K parent selection
- [ ] Test compact capped-frontier storage
- [ ] Test spatial chunk/cube ordering
- [ ] Test unique voxel-cube corner reuse if measurements support it
- [ ] Test persistent two-depth sampling session
- [ ] Test bounded envelope prefetch and rolling pins if measurements support them
- [ ] Test fixed caps 28, 24, and 20 after result-neutral work
- [ ] Test adaptive cap escalation if fixed-cap results support it
- [ ] Update specs, code-structure docs, changelog, status, and task log
- [ ] Run final focused tests and consistency review
- [ ] Request approval and run retained-final benchmark repetitions

Representative benchmarks always require explicit user approval immediately
before invocation and must reuse the exact approved command and cache path.
