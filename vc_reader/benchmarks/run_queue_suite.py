#!/usr/bin/env python3
"""Orchestrate native queue measurements; all timed work is in C++."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queues', default='boost,rigtorp,moodycamel')
    parser.add_argument('--producers', default='1,8,32,128,192,384,1920')
    parser.add_argument('--consumers', type=int, default=16)
    parser.add_argument('--capacity', type=int, default=4096)
    parser.add_argument('--seconds', type=float, default=.5)
    parser.add_argument('--timeout', type=float, default=28)
    parser.add_argument('--build-type', default='unspecified')
    args = parser.parse_args()
    queues = args.queues.split(',')
    producers = [int(x) for x in args.producers.split(',')]
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    metadata = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    build_type=args.build_type, system=platform.platform(),
                    available_cpus=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
                    queues=queues, producers=producers, consumers=args.consumers,
                    target_capacity=args.capacity, seconds_per_window=args.seconds,
                    native_timeout_seconds=args.timeout)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for queue in queues:
        run = subprocess.run([str(binary), queue, 'check'], capture_output=True, text=True, timeout=args.timeout)
        (args.output / f'check-{queue}.log').write_text(run.stdout + run.stderr)
        run.check_returncode()
    rows = []
    for mode in ('transfer', 'tickets'):
        for index, count in enumerate(producers):
            # Rotate queue order; never overlap cases or compilation.
            order = queues[index % len(queues):] + queues[:index % len(queues)]
            for queue in order:
                command = [str(binary), queue, mode, str(count), str(args.consumers),
                           str(args.capacity), str(args.seconds)]
                name = f'{mode}-{count}-{queue}'
                try:
                    run = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                except subprocess.TimeoutExpired as error:
                    (args.output / f'{name}.timeout.json').write_text(json.dumps(dict(command=command, timeout=args.timeout)))
                    raise RuntimeError(f'Native queue case exceeded {args.timeout}s: {name}') from error
                (args.output / f'{name}.log').write_text(run.stdout + run.stderr)
                run.check_returncode()
                row = json.loads(run.stdout)
                if row['errors']:
                    raise RuntimeError(f'Queue validation failed: {name}')
                rows.append(row)
                (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
                print('QUEUE_CASE', name, round(row['transfers_per_second']['median']), flush=True)
    print('QUEUE_SUITE_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
