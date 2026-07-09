# Current Task

Fix 2D fiber-strip prefetch so VC3D is only used to calculate required chunks,
while all cache checks, missing markers, downloads, temporary files, atomic
renames, retries, and parallelism are handled in Python.

- Revert accidental VC3D changes that implement chunk downloading/fetching or
  persistent-cache writing. Keep only VC3D functionality needed to calculate
  chunk dependencies for the coordinate envelopes.
- `.empty` missing markers are zero-byte empty marker files. They do not need
  temporary files; create them directly only for definitive missing chunks.
- Change VC3D's own persistent-cache `.empty` writer to also write zero-byte
  files. VC3D already treats `.empty` markers by existence only, so this should
  not change read semantics.
- The Python prefetcher must download required base-volume chunks itself.
- Python chunk downloads must write into a unique temporary file in the target
  cache directory, then atomically rename to the final chunk path only after the
  complete payload has been written.
- Temporary file names must be unique enough to avoid collisions between worker
  processes/threads and stale concurrent runs.
- Run up to 16 parallel Python downloads.
- Prefetch progress must include ETA based on observed download rate plus the
  observed cache-hit/known-missing/download-needed ratio. While dependency
  generation is still running, estimate remaining download work by assuming the
  observed ratio continues for not-yet-processed samples.
- Do not use VC3D image sampling or VC3D chunk fetching as the production
  prefetch download path.
- Remove Python prefetch fallbacks that call VC3D `prefetch_chunk`,
  `read_chunk`, `sample_coords`, or generic store indexing for production
  remote VC3D prefetch.
- Ensure the VC3D dependency function returns each required chunk with explicit
  path/format information for Python to consume:
  remote chunk source/path, final persistent cache data path/extension,
  expected cache payload format, and `.empty` marker path.
- Python must not reconstruct VC3D remote/cache paths for VC3D chunk requests.
  It should use the paths and format metadata returned from VC3D.
- Python must write the same cache payload format that VC3D expects for the
  current zarr codec. If that cannot be derived safely, fail with a clear error
  instead of silently using the wrong format.
- Current implementation scope is uncompressed direct-source zarr chunks only:
  for those chunks the remote payload is the same byte payload VC3D expects in
  the persistent `.bin` cache file. Compressed, filtered, sharded, byte-swapped,
  or otherwise non-direct chunks must fail clearly until explicit codec support
  is added.
- Keep the existing dependency discovery behavior: generate configured
  augmentation-safe coordinate envelopes, ask VC3D which chunks those envelopes
  need, deduplicate globally, then download/cache only the missing chunks.
- Preserve deterministic sample selection and training/augment-vis loading
  behavior.
- If CP-local strip construction hits invalid Lasagna normal data such as an
  in-bounds `grad_mag == 0` line-window point, prefetch and training should
  skip that deterministic sample and continue. Infrastructure/programming
  failures still stop the run instead of being hidden.
