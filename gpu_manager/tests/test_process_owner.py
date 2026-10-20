import os
import subprocess
import sys

from villa_gpu_manager.process_owner import ProcessOwner
from villa_gpu_manager.processes import process_identity, process_state
from villa_gpu_manager.client import Client
from test_exclusive import memory_client
from test_manager import engine


def test_owner_ack_and_eof_cleanup():
    owner = ProcessOwner()
    worker = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                              start_new_session=True)
    try:
        identity = process_identity(worker.pid)
        owner.track([identity])
        assert process_state(identity, os.getuid()) == 'alive'
        owner.close()
        worker.wait(timeout=5)
        assert process_state(identity, os.getuid()) == 'dead'
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        if owner.process.poll() is None:
            owner.close()


def test_two_sibling_roots_do_not_claim_coordinator():
    e = engine()
    clients = [memory_client(e)('unused', registration_root=dict(pid=i,start_us=1,boot_us=1))
               for i in (100,101)]
    request = dict(max_cpu_cores=2,max_gpus=1,min_vram_bytes_per_gpu=10,
                   max_vram_bytes_per_gpu=20,allowed_gpu_uuids=['GPU-a'],
                   priority=100,exclusive=False)
    for client in clients:
        client.connect()
        client.register('sibling', request)
    assert {t['root']['pid'] for t in e.tasks.values()} == {100,101}
    assert sum(t['granted']['cpu_cores'] for t in e.tasks.values()) <= 4


def test_retire_joined_worker_before_reusing_owner():
    owner = ProcessOwner()
    try:
        for _ in range(2):
            worker = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                                      start_new_session=True)
            identity = process_identity(worker.pid)
            owner.track([identity])
            worker.terminate()
            worker.wait(timeout=5)
            owner.retire([identity])
    finally:
        owner.close()
