"""CPU-only lifecycle demonstration. Replace Worker callbacks with your workload.

Never report release until all CUDA-owning child processes have exited/joined.
This example creates NO GPU workers and claims NO GPU memory use for real work.
"""

import argparse
import time

from villa_gpu_manager import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("socket")
    args = parser.parse_args()
    request = {"max_cpu_cores": 1, "max_gpus": 1,
               "min_vram_bytes_per_gpu": 1024**3, "max_vram_bytes_per_gpu": 1024**3,
               "allowed_gpu_uuids": [], "priority": 0, "exclusive": False}
    with Client(args.socket) as client:
        client.register("metadata-only-example", request)
        for _ in range(3):
            state = client.poll()
            print(state["phase"], state["granted"], flush=True)
            if state["phase"] == "initialize":
                client.report("applied", state["granted"])
            elif state["phase"] == "release":
                client.report("released", {"cpu_cores": 0, "gpus": {}})
            time.sleep(1)
        state = client.pause()
        if state["granted"] != {"cpu_cores": 0, "gpus": {}}:
            client.report("released", {"cpu_cores": 0, "gpus": {}})
        client.finish()


if __name__ == "__main__":
    main()
