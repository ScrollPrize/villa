# Task: robust auto-download cache and worker control

Fix Fiber/Lasagna automatic OME-Zarr download failing when an interrupted
download leaves an empty or malformed `.dl_cache/<level>.noremote.json` file.
The negative-remote cache is advisory and corruption must not abort inference.

Also expose a CLI option that controls the downloader's parallel transfer
worker count independently of inference prefetch and pyramid workers. Preserve
the downloader's current default of 64 workers.

Validate the fix, document the option, and commit the completed changes.
