# Plan: Remove Unpublished Fiber Version 2

## Readers And Validators

1. Accept only top-level `vc3d_fiber` versions 1 and 3 in VC3D, C++ core
   atlas/tracer/probe readers, Python training readers, and sync validation.
2. Keep numeric CP arrays for version 1 and object CPs for version 3.
3. Remove version-2 load migration and the pre-v3 segment metadata parser
   branches. Version-3 descriptors continue to require metadata version 3 and
   tracer version 2.
4. Keep writers unchanged: they already emit version 3 only.

## Sync

1. Keep the new conservative version-3 span merger unchanged.
2. Retain the legacy geometry merger only for version 1.
3. Treat version-2 files as invalid/unloadable and route file conflicts to
   manual handling rather than attempting a merge.

## Tests

1. Replace migration tests with explicit version-2 rejection tests.
2. Retain version-1 and version-3 parsing, round-trip, merge, and strict-schema
   coverage.
3. Run the focused C++ line-annotation/atlas tests, Python fiber parser tests,
   and sync merge/helper tests.

## Spec Update

- State that only file versions 1 and 3 are supported and that pre-v3 segment
  metadata is unsupported.

## Docs Updates

- Remove version-2 compatibility and migration descriptions from fiber docs.

## Changelog

- Record removal of the unpublished version-2 format.
