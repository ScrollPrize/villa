# Task

Prevent inference scratch files from surviving into exported cache views.

Requirements:

- retain an explicit cleanup scan for legacy `.live_tmp/session-*` directories;
- run cleanup only after exclusive cache-writer ownership is acquired;
- use anonymous/unlinked file-backed mappings for new reusable result-slot
  backing where the platform supports it;
- keep a safe named-file fallback for platforms/filesystems without
  `O_TMPFILE`;
- never remove unrelated files, symlinks, or published Zarr data;
- validate with focused demand-inference tests and syntax checks.
