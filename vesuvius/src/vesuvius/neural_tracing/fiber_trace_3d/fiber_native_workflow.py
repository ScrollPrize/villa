"""Portable native workflow launcher with immutable-input provenance."""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from lasagna.inference_provenance import atomic_write, sha256_file
from .constraint_extraction import resolve_native_binary


def resolve_command(case, config):
    return [resolve_native_binary(case['binary'], config),*map(str,case.get('args',[]))]


def run(config):
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    commands=[(case,resolve_command(case,config)) for case in config['commands']]
    inputs={str(p):sha256_file(p) for p in config.get('input_files',[])}
    references={str(p):sha256_file(p) for p in sorted(Path(config['reference_dir']).glob('*.json'))} if config.get('reference_dir') else {}
    records=[]
    for case,command in commands:
        name=case['name']
        if not name or Path(name).name!=name or name in {'.','..'}:raise ValueError('Case name must be a simple filename')
        existing=[root/f'{name}.{suffix}' for suffix in ('log','run.json')]
        if any(p.exists() for p in existing):
            archive=root/'attempts'/(name+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            archive.mkdir(parents=True)
            for p in existing:
                if p.exists():p.rename(archive/p.name)
        record=dict(command=command,binary_sha256=sha256_file(command[0]),inputs=inputs,references=references,
                    started_utc=datetime.now(timezone.utc).isoformat())
        started=time.monotonic()
        with (root/f'{name}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode,wall_seconds=time.monotonic()-started)
        atomic_write(root/f'{name}.run.json',record,durable=False)
        result.check_returncode()
        if any(sha256_file(p)!=digest for p,digest in inputs.items()):raise ValueError('Native inputs changed during execution')
        records.append(record)
    return records
