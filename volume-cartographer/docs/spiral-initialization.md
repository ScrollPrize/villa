# Spiral initialization and input working copies

Initialize Fit and Rebuild Fit retain the service editing lease and local drafts.
Only actions that leave the workspace run the session-exit guard and release
editing ownership. The service replays accepted input revisions after a rebuild.

Creating a fiber or patch editor working copy runs in Qt's worker pool. The
Spiral panel's fixed status area shows an activity indicator, elapsed time, and
copied file count and MiB. Connection-time catalog preparation has its own stages:
reading the catalog, requesting editing access while the service prepares or
checks input snapshots, and loading the input list. These stages also appear in
Logs. The preparation indicator is indeterminate because the service does not
report a completed/total count; elapsed time measures the wait, not server progress.
Concurrent requests for the same source share a copy. Disconnecting or releasing
the workspace cancels pending copies and prevents their callbacks from reopening
editors. Workers own their temporary directories independently of the UI; cleanup
also runs in the worker pool. Copying preserves the existing recursive file order,
file contents, and rejection of symbolic links.

Fiber-editor adoption remains on the UI thread. If fibers change while the
initialization copy runs, adoption is rejected instead of replacing newer edits;
reconnect to make a fresh copy. Fiber parsing and editor refresh can still cause a
separate pause after copying; this change does not move the controller's mutable
UI state into a worker.

## Build and validation

From `volume-cartographer/`, using the existing configured build and dependencies:

```sh
AGENTS_AGENT_MODE=1 ninja -C build VC3D test_spiral_input_workflow -j2
python3 apps/VC3D/test/check_spiral_initialize_ownership.py
QT_QPA_PLATFORM=offscreen \
SPIRAL_TEST_PYTHON="$PWD/../spiral-fitting/.venv/bin/python" \
SPIRAL_TEST_COPY_SOURCE=/path/to/real/dataset/fibers \
build/bin/test_spiral_input_workflow
build/bin/VC3D
```

The ownership harness executes the production button handler and checks that
initialization and rebuild retain the lease. The copy tests verify bytes from a
real directory, completion on the UI thread, UI heartbeat activity, coalescing,
reuse, cancellation, and error delivery. They write only to temporary directories.
The complete workflow suite also starts a local test service. Qt's `QFile::copy`
requires working temporary-file support; some execution sandboxes reject its
output-file creation even when ordinary file writes work.

Validated with the existing RelWithDebInfo build on Ubuntu arm64, Qt 6.4.2, using
`/mnt/bigpc/spiral_dataset_working/fibers`: 694 files, approximately 307 MiB.
An observed copy completed in 15.6 seconds while launch took less than 1 ms and
1,539 UI heartbeat callbacks ran. This is a responsiveness check, not a disk
throughput benchmark or a measurement of complete model initialization.
