# FM-A01 — Artifact materialization bypasses backing-store output protection

Baseline: HEAD `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4` plus the reviewed uncommitted post-fusion fixes. High-priority correctness candidate, source-traced; no destructive runtime probe performed. Older omission: `0c7e4e521:src/ObjectStoreSession.cpp` has the same path.

`ObjectStoreSession::materializeArtifacts` obtains a selected object's manifest and writes `destination/file.name` through private `writeArtifact`. The local branch invokes `detail::writeArtifactFile` without `ObjectStore::validateExportDestination`. The latter streams verified content into `vcr::AtomicFile` and calls `publish(false)`. In `vc_reader/src/cpp/atomic_file.hpp`, false means no durability sync, **not** no-overwrite: publication replaces the destination path.

Concrete valid-input counterexample: a store at `/tmp/case/objects.vcobj` contains an artifact named `objects.vcobj` (allowed by `Artifacts.cpp::checkName`). Call session `materializeArtifacts` with destination `/tmp/case`. The validated artifact bytes replace the attached database pathname. Its existing open connection may still reference the old inode, but subsequent opens see artifact bytes, not the database. This is accidental output collision, not an invalid reference or hostile SQL input. Referenced dependency database paths, packed containers and artifact sidecars need the same protection.

Sibling boundary trace:

- `ObjectStoreSession::artifacts` correctly resolves current/numbered/exact selected metadata; that identity correctness does not protect the filesystem destination.
- Python session `materialize_artifacts` delegates the unsafe local materialization path. `artifact_file` also reaches `writeArtifact` for its generated cache target, so user-controlled cache roots should pass the same source-closure check.
- `geometry/GeometryExport.cpp` calls `reader.validateExportDestination` before output. `SourcePaths.cpp::ObjectStore::validateExportDestination` already checks the resolved root/dependency closure, packed containers, SQLite sidecars and artifact directories, including equivalent paths.
- Hosted `HostedStore::fetchArtifact` and `HostedAttachment::writeArtifact` also use the atomic byte writer, but a remote server's backing paths are not client-local files; do not add guessed server-path policy on the client. Protect relevant local attached backing stores where the owning layer knows them.
- `Artifacts.cpp::ArtifactExport` uses the same byte writer for an intentionally created destination sidecar during a protected copy workflow. Do not globally forbid that legitimate internal destination.

Recommendation: apply the existing closure-aware validator at the local attachment/store materialization owner before any destination creation/publication; centralize rather than implement basename/suffix checks in Python. Check each final output target (and decide whether to preflight all targets before writing any). Preserve normal overwrite/cached-repair behavior outside the backing closure.

Required regression cases: same database filename under its parent, referenced dependency, packed container, SQLite sidecars, artifact sidecar, ordinary safe output, current/exact manifest selection. Test only isolated temporary stores. This review claims source evidence, not execution or recovery of overwritten user data.
