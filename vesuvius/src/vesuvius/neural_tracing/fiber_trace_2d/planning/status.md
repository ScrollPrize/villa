# Status: cache decoded fiberlet graph chunks

- [x] Record the task
- [x] Read the storage specification and current cache/graph implementation
- [x] Draft implementation, test, spec, docs, and changelog plan
- [x] Independently review the plan against task and specifications
- [x] Establish a cache-warm replay baseline
- [x] Add typed decoded payloads to `ChunkCache`
- [x] Add parse-once dataset validation/materialization
- [x] Materialize indexed anchor/prefix/route cache entries
- [x] Convert graph traversal to chunk payload leases and neighbor prefetch
- [x] Keep route payloads out of lookahead
- [x] Add focused regression tests
- [x] Build with 32 jobs and run focused tests
- [x] Measure cache-warm replay after the change and verify identical results
- [x] Update specifications, docs, changelog, and task log
