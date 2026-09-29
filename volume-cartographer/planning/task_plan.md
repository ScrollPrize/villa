# Plan

1. Return extract_inner_chunk() directly from the sharded branch of
   decode_chunk_from_storage_object(): extraction already decompresses and
   converts byte order. Keep the unsharded path unchanged.
2. Document the extraction method's decoded/native-byte-order contract.
3. Extend the existing sharded cache tests to use production compression codecs,
   verify exact bytes, missing chunks, sibling chunks and disk-mirror reopen.
   Pass the production codec explicitly to the fixture writer, then reopen with
   the registry (create(registry) only inspects outer codecs on current main).
   Add a non-native-endian contract regression with stored-order bytes and an
   outer bytes codec to activate the existing swap path. Nested-endian support
   is a separate pre-existing issue, outside this fix.
4. Demonstrate the regression on unpatched main, build with 32 jobs using existing
   dependencies, and run focused Zarr/cache tests. Seek real-scroll validation;
   clearly disclose if only synthetic coverage is available.
5. Obtain independent plan/code review and preview the PR before publishing.

## Specification Updates

No new behavior contract: restore correct decoding under the existing shared
cache architecture. No scheduling, mirror format or shard writer changes.

## Documentation Updates

Clarify extract_inner_chunk() in its public header and record validation here.

## Testing And Validation

Use the production codec registry and storage-object/cache path, not just direct
chunk reads. Include zstd/gzip and uncompressed data. Compare exact decoded data
and verify the disk mirror retains encoded shard bytes.

## Changelog Update

Add one dated entry on completion.
