# SFTP read-ahead

The hub is a mutable SQLite file, protected by an exclusive remote process lock.
Reading its pages without that lock would allow mixed versions of the database,
including uncommitted writes. Immutable geometry rows do not make their SQLite
pages immutable. Removing this lock from transfers would require a separate
snapshot/publication protocol or a server-side object-store service.

The replica already stages incoming geometry before taking its local write lock.
The immediate bottleneck was the VFS submitting each SQLite page separately:
the SFTP client's existing pipelining only helps when given a larger read.
The VFS now caches eight 64 KiB aligned blocks, using those pipelined reads.
Several blocks accommodate interleaved index and blob-overflow accesses. Cache
entries are discarded on a new locked connection generation or any write or
truncate through that connection, including another handle and rollback writes.
Failed/inactive sessions cannot serve cache hits. Journals and reads smaller than
512 bytes bypass read-ahead, keeping idle change-counter checks inexpensive.
No file format, numerical, transaction, or server setup changes are required.

## Reproduce

With the existing dependencies installed, the storage-only test build is:

```sh
AGENTS_AGENT_MODE=1 cmake -S vc_object_store -B /tmp/vc-sftp-readahead-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVC_OBJECT_STORE_BUILD_ADAPTERS=OFF -DVC_OBJECT_STORE_BUILD_TOOLS=OFF \
  -DVC_OBJECT_STORE_BUILD_LINKING=OFF -DVC_OBJECT_STORE_BUILD_TESTS=ON
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-sftp-readahead-build -j 4
AGENTS_AGENT_MODE=1 ctest --test-dir /tmp/vc-sftp-readahead-build --output-on-failure -j 4
/tmp/vc-sftp-readahead-build/bin/test_object_store_sftp --real-store /path/to/fibers.vcobj
/tmp/vc-sftp-readahead-build/bin/test_object_store_sftp --benchmark-pull /path/to/fibers.vcobj 16 3 20
```

The benchmark arguments are source store, fiber count, repetitions and fixture
reply delay in milliseconds. It reads the source without modifying it, selects
fibers in UUID order, and copies their current geometry to a temporary hub.
Each timed pull installs those fibers into a fresh replica with a cold hub
handle. Setup and subsequent byte-for-byte array verification are not timed.
The fixture counts SFTP replies and data bytes. It delays each forwarded server
output batch; this is a repeatable latency fixture, not a bandwidth-limited WAN
emulator. Like the process-lock tests, it requires Linux and OpenSSH sftp-server,
but no sshd or credentials. The implementation itself uses portable C++.

## Measurements

Measured on Linux arm64, GCC 13.3, Release, using the command above with
`/tmp/vc-hub-lock-real.vcobj`: 16 real fibers, 1,873,352 geometry-array bytes,
three pulls per version, 20 ms fixture reply delay. The baseline binary used
the same benchmark and the original page-at-a-time VFS.

| Metric | Before | After |
| --- | ---: | ---: |
| Mean pull time | 11.411 s | 2.455 s |
| Min / median / max | 11.408 / 11.410 / 11.417 s | 2.442 / 2.460 / 2.461 s |
| SFTP data replies per pull | 536 | 101 |
| Total SFTP replies per pull | 553 | 118 |
| Data bytes received per pull | 2,166,249 | 3,080,472 |

This workload is 4.65 times faster, with 81% fewer data replies and 42% more
bytes transferred. Zero-delay control runs (three per version) averaged 0.058 s
before and 0.051 s after; at that scale local scheduling noise is substantial.
Read-ahead can cost bandwidth on sparse/random access, and does not eliminate
lock contention on bandwidth-bound transfers. Cache storage is capped at
512 KiB per open database handle.

A separate wall-clock I/O profile used:

```sh
strace -f -c -w -o /tmp/sftp-profile.txt \
  /tmp/vc-sftp-readahead-build/bin/test_object_store_sftp \
  --benchmark-pull /tmp/vc-hub-lock-real.vcobj 16 1 20
```

Across the client and fixture processes, `ppoll` dominated waiting (36.43 s
summed before, 8.68 s after); fixture sleeps dropped from 11.17 s to 2.39 s.
These overlapping process totals include setup and are not pull elapsed times.
Timing results in the table were collected without strace.

The regression suite checks that 32 sequential 4 KiB VFS reads need at most
four SFTP data replies, plus unaligned/cross-block reads, eviction, EOF zero
filling, write/truncate/regrow invalidation across handles, new lock generations,
and failed/inactive scopes. Existing tests cover hot journals, lost replies,
server death/deadlines, other writers, and local editing during slow pulls.
