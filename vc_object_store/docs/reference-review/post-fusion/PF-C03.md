# PF-C03: Loaded tool pins do not automatically advance — not a confirmed defect

`StoreSurfaces::update` does not repin `ToolObjects::read_` from its receipt. Repeating a write or publishing artifacts through that loaded entry continues to guard its originally read reference. This initially looked like missing editor receipt handling.

Astra rejected automatic repinning as an established correctness fix: ToolObjects explicitly tracks the versions a command-line tool read, and publishArtifacts is documented to expect that read state. Automatically advancing could certify artifacts still computed from the old geometry. The production ObjAlphaCompRefinement caller performs one update; no repeated-save promise was established.

If a reusable editor API is later required, it needs an explicit accepted-write/reload contract that refreshes points, payload/channel identity and mapping baseline together. Receipt-only silent repinning is not justified. Keep this candidate/disposition so future reviews do not repeatedly count the same intentional contract as a new bug.
