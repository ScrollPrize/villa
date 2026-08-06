# Plan: manager no-prefetch download behavior

## Implementation

1. Make both backend command builders accept whether downloads are disabled.
2. Derive that value from the manager prefetch choice: prefetch enabled means
   backend `--no-download`; `--no-prefetch` means backend auto-download.
3. For an empty manager cache, initialize only the local `_download` source
   descriptor (no remote scan or chunks) through a public downloader helper so
   backend crop-aware auto-download can bootstrap itself. Forward the manager
   download-worker value to that backend path.
4. Preserve detached prefetch ordering, resolved argv provenance, explicit
   backend arguments, and direct inference defaults.

## Tests

5. Cover the Fiber/Lasagna × prefetch/no-prefetch argv matrix, empty-cache
   source initialization, worker forwarding, explicit `--no-download`
   precedence, and CLI dispatch.
6. Run focused manager tests, compilation, and diff checks.

## Spec update

Define the relationship between manager prefetch and backend download mode.

## Docs updates

Clarify that `--no-prefetch` delegates downloading to inference rather than
requiring a completely local input, and that this is crop-aware backend
downloading during the inference lifecycle.

## Changelog

Add a short manager download-mode entry.
