# Task plan

## Discovery

- Confirm where cache status and Z-scroll sensitivity widgets are created and
  updated.
- Preserve the existing `ViewerManager` cache-stat and sensitivity signals.

## Implementation

1. Format persistent-cache storage as `RAM X/Y disk X/Y GiB`, with the unit
   following both values once.
2. Remove the separate Z-sensitivity permanent widget from `CWindow`.
3. Retain the latest cache fields in `CWindow` and render cache fields plus
   `Z sens: N.N` through one status-label update helper.
4. Keep the Z-scroll tooltip on the merged status label.

## Testing

- Extend the focused cache-status formatter test for the shared-unit grammar.
- Build and run `test_download_queue_stats`.
- Build VC3D to validate the merged widget path.

## Spec update

- Correct the documented storage syntax and require a single permanent cache /
  Z-sensitivity label in the main window.

## Docs updates

- Update `docs/remote_file_cache.md` examples to the corrected unit placement.

## Changelog update

- Add a concise entry for the corrected and merged status display.
