# ext4 allocation stall on md0 — 2026-09-28

## Findings

1. **Confirmed:** the sampled writeback workers spend essentially all sampled
   CPU time in ext4 block-allocation group selection, not device transfer.
2. **Confirmed:** the exact running kernel's compiled allocator lacks the
   complex-scan fallback added by upstream commit
   `1f6bc02f18489b9c9ea39b068d0695fb0e4567e9`.
3. **Confirmed:** the captured workload executes that unpatched stripe-aligned
   branch. Two sampled call chains contain its exact return instruction address.
4. **Strong diagnosis, not complete live-state proof:** this is the pathological
   repeated-group allocation search described by the fix. The profile does not
   capture allocation-context pointers, selected group IDs, requested lengths,
   inode IDs or scan outcomes; it cannot prove the same request/group repeats
   indefinitely or identify the specific scratch file responsible.

No kernel/filesystem settings changed. No fsck, unmount, reboot, global sync,
package install, or workload restart was performed during this investigation.

## Host and incident

- Ubuntu 22.04, running `6.5.0-1018-oracle`, build
  `#18~22.04.1-Ubuntu SMP Sat Feb 17 21:41:13 UTC 2024`.
- `/dev/md0` (major/minor 9:0), ext4 mounted at `/ephemeral`, host mount `rw`,
  `stripe=2048`. Sandbox has a separate read-only view; probe ran outside it.
- `df`: 56T total, 47T used, 6.2T available, 89% used. This does not measure
  free-space fragmentation or prove on-disk corruption.
- 07:04 UTC: kernel workers 788679 and 1067978 named `flush-9:0` were CPU-busy.
  Journal PID 6775: `jbd2_journal_wait_updates ->
  jbd2_journal_commit_transaction -> kjournald2`. This waits for outstanding
  transaction updates, not a disk completion at the sampled stack location.
- Other jobs and ncdu blocked in `wait_transaction_locked`.
- md0 completed read/write counters unchanged over a 2-second sample, seven
  outstanding requests. Insufficient by itself to diagnose device health.
- Isolated write probe blocked creating a unique file for 10 seconds, before
  any 4 KiB write or fsync. SIGKILL sent; completion of cleanup was not verified.

## Perf evidence

User captured `/tmp/ephemeral-kworkers.data` at 07:12:38 UTC with:

```bash
sudo perf record -e cpu-clock -F 49 -g -p 788679,1067978 \
  -o /tmp/ephemeral-kworkers.data -- sleep 10
sudo cat /proc/kallsyms > /tmp/ephemeral-kallsyms.txt
```

Read-only analysis (force handles sandbox ownership mapping):

```bash
perf report -f -i /tmp/ephemeral-kworkers.data \
  --kallsyms /tmp/ephemeral-kallsyms.txt --stdio --no-children -g none --percent-limit 0.1
perf script -f -i /tmp/ephemeral-kworkers.data \
  --kallsyms /tmp/ephemeral-kallsyms.txt
```

505 samples, no lost samples. Percentages are shares of these samples, not
whole-machine utilization. Worker 1067978 contributes 97.03% of samples;
788679 contributes 2.97% (earlier `ps` CPU values were lifetime averages).

| Function | Self sample share, both workers combined |
|---|---:|
| ext4_mb_good_group | 49.51% |
| ext4_get_group_info | 34.06% |
| ext4_mb_find_good_group_avg_frag_lists | 15.65% |

Path: writeback -> ext4_writepages -> mpage_map_and_submit_extent ->
ext4_map_blocks -> ext4_ext_map_blocks -> ext4_mb_new_blocks ->
ext4_mb_regular_allocator -> best-available group selection.

Samples at monotonic times 860078.462135 and 860085.033558 include:

```text
ffffffffb09b1f80 ext4_mb_scan_aligned+0xf0
ffffffffb09b33ad ext4_mb_regular_allocator+0x71d
```

## Exact binary verification

Downloaded (not installed) official Ubuntu package:

`https://archive.ubuntu.com/ubuntu/pool/main/l/linux-signed-oracle-6.5/linux-image-6.5.0-1018-oracle_6.5.0-1018.18~22.04.1_amd64.deb`

- Local `/tmp/oracle-1018-image.deb` SHA256:
  `e0713b514dfbec00254cbd5da0fb05c0c511dd3d1a1b9615fa1d297504ca82d3`,
  exactly matching configured apt metadata.
- Extracted with dpkg-deb to `/tmp/oracle-1018-inspect` (not installed).
- Upstream Linux v6.5 `scripts/extract-vmlinux` decompressed the image into
  `/tmp/oracle-1018-vmlinux`.
- ELF GNU build ID `ca0ebe8715b038e0a0fd60e49cf9a2b5168dc79a` equals the running
  kernel build ID recorded by `perf buildid-list`.
- Runtime `_text=ffffffffb0400000`, ELF `.text=ffffffff81000000`;
  KASLR relocation is `0x2f400000`.

Relevant ELF disassembly:

```text
ffffffff815b2f16: je   ffffffff815b33a1   # stripe-aligned branch
ffffffff815b2f23: call ffffffff815b2300   # complex_scan_group
ffffffff815b2f28: ...                   # common unlock/cleanup
...
ffffffff815b33a8: call ffffffff815b1e90   # scan_aligned
ffffffff815b33ad: jmp  ffffffff815b2f28   # skips complex fallback unconditionally
```

Thus the runtime sampled caller `b09b33ad` maps exactly to the unpatched jump.
This is stronger than guessing from kernel version or upstream base source.

## Upstream fix and interpretation

[Upstream commit](https://github.com/torvalds/linux/commit/1f6bc02f18489b9c9ea39b068d0695fb0e4567e9)
is titled **ext4: fallback to complex scan if aligned scan doesn't work**.
It describes repeatedly selecting the same group after an unsuccessful
stripe-aligned scan, potentially consuming hours on large filesystems, and
adds `ext4_mb_complex_scan_group` when `ac_status == AC_STATUS_CONTINUE`.

Our binary instead unconditionally bypasses that fallback after aligned scan.
Mount stripe geometry, active unpatched branch, dominant repeated group scans,
journal wait, and absent sampled I/O completions all fit this failure mode.
The exact Ubuntu source endpoint returned 403, so patch status was established
from the build-ID-matched compiled kernel instead.

## Remaining limits and safe next steps

- Definitive repetition proof would require bounded root tracing of allocation
  context/group and scan status on the affected workers (not merely another
  CPU profile). No such instrumentation has been installed or run.
- Do not claim an infinite loop or predict time-to-recovery from these samples.
- Do not attribute the exact inode or first triggering operation to accumulator
  spill without tracing it; these are buffered writeback stacks, not proof of
  the direct-I/O spill call path.
- A durable remedy should use a kernel whose actual allocator includes this
  fix. Kernel upgrade/reboot and any live workaround require separate approval;
  availability of a newer kernel alone is not patch verification.
- Keep completed Classic/low-res results and inference cache intact. Full-res
  remained stalled; extraction has not completed.
