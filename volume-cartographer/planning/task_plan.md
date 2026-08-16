# Task plan

## Design

1. Extend the chunk-fetcher persistence contract so a fetched source payload
   can be written before and independently of decoding. Store maintenance
   payloads under one generic exact-source extension and teach normal reads to
   decode that representation. Keep ordinary interactive cache-write policy
   unchanged: decoded `.bin`, cache-compressed `.zst`, and existing native
   encoded formats remain valid, readable representations.
2. Add a persistence-only chunk request API to `ChunkCache`. It must support:
   - ensure mode, which accepts an existing persistent entry;
   - refresh mode, which fetches the source again and atomically replaces the
     persistent entry;
   - blocking result/status reporting for bounded background workflows.
3. Keep per-key maintenance request/completion state separate from decoded
   entry state. Both consumers join one source-transfer registry keyed by
   source/chunk. A refresh never hides, invalidates, or waits on resident decoded
   data. Joining works in both orders: persistence joins an existing interactive
   transfer, and an interactive miss joins and promotes an existing maintenance
   transfer. Both orders perform one source read.
4. Add a third maintenance scheduler class below interactive and ordinary
   background work. Maintenance runs only when neither higher class is pending.
   If an interactive or normal background consumer joins the same keyed
   transfer, reprioritize that transfer in place into the higher class.
5. On a shared source result, give decode and persistence independent completion
   paths over the same immutable fetched payload. Rendering may publish decoded
   data without waiting for disk, while maintenance completes only after its
   atomic disk write commits. Decode only when a decoded consumer exists.
   Persistence-only completion must not insert bytes into the decoded LRU or
   decoded-byte budget.
6. Ensure persistent replacement removes stale `.bin`, `.zst`, `.c3d`, exact
   source, and `.empty` alternatives,
   remains atomic, participates in the existing persistent-disk budget, and
   reports completion only after the write commits. Found source payloads become
   exact-source entries; authoritative source missing results become `.empty`;
   HTTP/I/O failures preserve the prior entry and report failure. Without a
   decode, encoded all-fill chunks are counted as found data.
7. Change Open Data prefill to use the normal Volume's shared cache handle and
   persistence-only ensure requests. Keep one bounded producer and background
   priority; retain cancellation, progress, completion markers, and error
   accounting.
8. Change Settings "Redownload cache" to use persistence-only refresh requests
   on the current Volume's shared cache. Remove its private service, custom
   source-read workers, and decode/recompression path. Refresh only keys already
   represented on disk and recognize `.bin`, `.zst`, `.c3d`, `.empty`, and the
   exact-source extension. Remove compression/quantization semantics from
   redownload and relabel its worker control as compression-only; the separate
   existing-cache compression action remains responsible for offline
   recompression.
9. Remove both Volume-level isolated-cache factories and their process-budget
   injection. Keep lower-level standalone cache factories for genuinely
   separate processes, batch tools, and tests; an explicit isolated service owns
   its complete budget and shares nothing with the process service.
10. Audit all in-process `prefetchChunks()` users and Lasagna channel samplers.
    Ordinary Volume-based sampling remains on the process cache. Make all
    in-process Lasagna channel arrays acquire canonical array identities from
    the process service, eliminating their private-service/shared-budget hybrid.
    Standalone executables naturally get their own process service.
11. Add the exact-source extension to persistent budget discovery, probing,
    alternate cleanup, invalidation, and redownload enumeration.

## Tests

- Add a fetcher fixture that counts source fetches and decode calls.
- Verify persistence-only ensure writes bytes and performs zero decodes.
- Verify persistence-only refresh replaces disk data and performs zero decodes.
- Verify an interactive request joining a queued persistence request causes one
  source fetch, one decode, and one persistent write.
- Verify persistence joining an existing interactive source read also performs
  one source fetch and completes independently after disk commit.
- Verify interactive priority promotes the shared pending source task ahead of
  unrelated background persistence work.
- Verify maintenance does not start while interactive or ordinary background
  source work remains pending.
- Verify persistent-only completion does not increase decoded RAM accounting.
- Verify exact encoded Zarr payloads round-trip through the persistent cache and
  legacy decoded/cache-compressed entries remain readable.
- Verify remote missing replaces stale alternatives with `.empty`, while HTTP
  and I/O failures preserve the previous persistent entry.
- Update Open Data prefill tests for shared-cache, no-decode behavior and marker
  completion.
- Build `test_chunk_cache`, `test_open_data_volume_prefill`, VC3D, and the Python
  extension; run the complete core CTest suite and `git diff --check`.

## Spec update

- Replace both isolated prefill/redownload clauses with persistence-only
  maintenance demand on the process source service.
- State that persistence-only work shares source fetches and scheduling with
  interactive work, has a third lowest-priority class unless the same transfer
  gains a higher-priority consumer, and never populates decoded RAM by itself.
- State that exact encoded source payload is the maintenance representation,
  ordinary cache-write policy is unchanged, and legacy persistent entries
  remain readable.
- Clarify that explicit isolated services are reserved for standalone, batch,
  low-level array, and test workloads, and never partially share only RAM
  accounting with the process service.

## Documentation updates

- Update `docs/remote_file_cache.md` with the decoded-prefetch versus
  persistence-only distinction, scheduler/deduplication behavior, encoded
  payload format, and legacy-read behavior.
- Update API documentation for the persistence-only request and removal of the
  Volume prefill/redownload construction overload.

## Changelog

- Record the shared persistence-only download path and removal of decoded
  prefill/redownload caches.
