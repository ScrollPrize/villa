# PF-C02: StoreSurfaces no-op bypasses loaded guards

`volume-cartographer/core/src/StoreSurfaces.cpp::update` obtains the loaded `PinnedObject` through `ToolObjects::writable`, adding its expected object reference and mapping operation to a checked request. It compares the edited surface to the old immutable geometry, but returns `pin.reference().version` when no geometry operation was added. The checked request is never submitted, so its loaded-reference, write-lineage and mapping guards are not evaluated.

Counterexample: load surface version 1, another writer changes its metadata/geometry or coordinate registration, then call update with the unchanged loaded points. The function returns the old version rather than reporting that its loaded selection/mapping is stale. A real geometry edit uses the shared checked submit and refuses the same stale pin.

Astra accepted a no-op validation contract gap, not stored-data corruption: the branch writes nothing, and the helper currently documents returning its loaded version when unchanged. The store contract requires applicable no-op guards/revalidation to remain meaningful. Any fix should submit the guard-only request through the ordinary shared API, preserving no-op ownership/version behavior instead of adding consumer-local freshness queries. Older omission; runtime verification pending.
