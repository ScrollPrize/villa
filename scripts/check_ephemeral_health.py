"""Small bounded filesystem probe; no filesystem repair or global sync."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if sys.argv[1:] == ['write-probe']:
    print('probe: creating unique 4 KiB file', flush=True)
    fd, name = tempfile.mkstemp(prefix='.fiber-fs-probe-', dir='/ephemeral/hendrik')
    print(f'probe: created {name}', flush=True)
    try:
        os.write(fd, b'\0' * 4096)
        print('probe: write complete; fsync next', flush=True)
        os.fsync(fd)
        print('probe: fsync complete', flush=True)
    finally:
        os.close(fd)
        os.unlink(name)
    print('probe: close/unlink complete', flush=True)
    raise SystemExit(0)

print('UTC', datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
for line in Path('/proc/mounts').read_text().splitlines():
    if line.split()[1] in ('/', '/ephemeral'):
        print('HOST MOUNT', line, flush=True)
rows = subprocess.check_output(['ps', '-eo', 'pid=,stat=,pcpu=,wchan:32,cmd='], text=True)
for line in rows.splitlines():
    fields = line.split()
    if len(fields) < 4 or not fields[0].isdigit():
        continue
    if fields[1].startswith('D') or ('kworker' in line and float(fields[2]) > 1):
        print('PROCESS', line, flush=True)
        try:
            print(Path(f'/proc/{fields[0]}/stack').read_text(), flush=True)
        except OSError as exc:
            print('stack unavailable:', exc, flush=True)
for iteration in range(2):
    print('DISKSTATS', datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
    for line in Path('/proc/diskstats').read_text().splitlines():
        if line.split()[2].startswith(('md', 'nvme')):
            print(line, flush=True)
    if not iteration:
        time.sleep(2)
probe = subprocess.Popen([sys.executable, '-u', __file__, 'write-probe'], start_new_session=True)
try:
    print('probe exit', probe.wait(timeout=10), flush=True)
except subprocess.TimeoutExpired:
    probe.kill()
    print(f'PROBE TIMED OUT: PID {probe.pid}; SIGKILL sent, may remain blocked in kernel I/O', flush=True)
