# PF-C05: Native coordinate frames collapse to nullopt in ToolObjects

`ToolObjects::noteRead` records an optional VolumeVoxels frame. `StoreSurfaces::load` supplies nullopt for native-coordinate loads. `ToolObjects::coordinatesFor(sources, nullopt)` compares only those optional target frames and returns the first source's retained coordinate binding when all are nullopt. Two native loads with different declared scale_factor values or unrelated source volumes are therefore treated as the same frame.

Both public headers promise differently framed loaded sources are rejected. First-source binding chooses the result declaration only after source compatibility; it does not establish that distinct native units are compatible. The source references and bindings are already available and must be resolved through the shared APIs, without fresh-head guesses or another consumer-local transform interpreter.

Astra accepted the public-helper contract gap but found limited current CLI exposure: inspected writers usually publish zero/one source or explicitly select an output volume. No shipped multi-source data corruption is claimed. Preserve explicit output-volume override semantics and decide the unbound case explicitly. The omission is present at 0c7e4e521. Runtime probe planned.
