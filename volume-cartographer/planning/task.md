# Task: unify VC3D's regular chunk cache

Replace VC3D's competing per-volume regular chunk caches with one process-wide
decoded chunk-cache service. Cache contents must be independent of the current
volume and remain available across volume switches, overlays, and switches back
to a previously viewed volume, subject only to the shared cache capacity and
explicit source invalidation.

Every cached chunk must be qualified by its volume source. Resolve and intern
the stable path/URI-based source identity only when a volume source is
registered, then use a compact numeric `VolumeSourceId` in all render-time cache
keys and lookups. Render-time key construction and hashing must not process or
store source strings.

This task applies only to regular decoded volume chunks. Surface image tiles,
surface geometry tiles, and other derived caches remain separate. Their raw
volume chunk reads should use the unified regular cache rather than own a
competing decoded-chunk store.

Do not add volume-switch cancellation in this task. Existing in-flight work may
finish, while newly requested views receive normal newest-view priority. Do not
discard decoded entries when the current volume changes or its last viewer
client is released.

Preserve decoded values, persistent-cache layout and compatibility, local and
remote volume behavior, and existing public sampling behavior. Re-run the
synthetic chunked-plane rendering benchmark before and after the change and do
not accept a meaningful hot-path regression.
