# Idle lifecycle and logical CPU inventory (2026-09-10)

User-requested policy: automatically exit without clients; count logical CPUs.
Wire fields remain unchanged. CPU capacity now represents logical processors;
automatic discovery uses psutil.cpu_count(logical=True). Explicit serve capacity
remains supported.

The CLI service exits after 30 continuously idle seconds. A connection or any
nonterminal task prevents exit; disconnected tasks require normal process-tree
retirement, not speculative release. Successful close removes the socket and
keeps the ownership journal. CPU-only inventory replacement on restart requires
all tasks terminal with empty held/granted resources and no manual override.

Validation (no GPUs):

```bash
PYTHONPATH=gpu_manager/src /home/hendrik/.venv_las/bin/python -m pytest -q gpu_manager/tests/test_idle.py gpu_manager/tests/test_bootstrap.py gpu_manager/tests/test_manager.py
```

43 passed in 1.16s. Covers grace/reset, disconnected live ownership, dead task
retirement, clean socket removal, safe/refused inventory correction, logical CPU
discovery and existing manager behavior. Existing process 3286773 was stopped
with its graceful SIGINT drain handler; no manager process remained afterwards.
The next automatic start discovers 224 logical CPUs. No training launched.

Follow-up: the first change missed the separate physical-core limit in
processes.nvidia_inventory. Corrected it to logical CPUs as well; regression
tests now cover 128/224 accepted and 225 rejected on a 112-core/224-thread host.

Further startup fix: an auto-started manager is a descendant of its initiating
client. Exclude the server and its helper processes from client-owned process
trees, including discovery, reports and journal recovery, using full process
identities. Otherwise a stopped calibration can retain GPUs solely because the
manager is still alive. Real task workers remain protected by normal ownership.

Full CPU suite on 2026-09-11: 107 passed in 28.25s:

```bash
GPU_MANAGER_SCHEMA_PYTHON=/usr/bin/python3 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest -q gpu_manager/tests
```
