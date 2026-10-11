"""Acquire a reference tifxyz prefix with verifiable local provenance."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
from urllib.parse import urlparse


def relative_key(key,prefix):
    if not key.startswith(prefix):raise ValueError('Object outside requested prefix')
    relative=key[len(prefix):]
    if not relative or relative.startswith('/') or any(x in ('','..','.') for x in relative.split('/')):
        raise ValueError('Unsafe object name')
    return Path(*PurePosixPath(relative).parts)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(cfg):
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    source=urlparse(cfg['source'])
    if source.scheme!='s3' or not source.netloc:raise ValueError('Expected S3 source')
    prefix=source.path.lstrip('/').rstrip('/')+'/'
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    client=boto3.client('s3',config=Config(max_pool_connections=16))
    if cfg.get('list_only',False):
        pages=list(client.get_paginator('list_objects_v2').paginate(Bucket=source.netloc,Prefix=prefix,Delimiter='/'))
        report=dict(source=cfg['source'],objects=[dict(key=o['Key'],bytes=o['Size']) for p in pages for o in p.get('Contents',[])],
            prefixes=[o['Prefix'] for p in pages for o in p.get('CommonPrefixes',[])])
        (root/'listing.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return report
    objects=[obj for page in client.get_paginator('list_objects_v2').paginate(Bucket=source.netloc,Prefix=prefix)
             for obj in page.get('Contents',[]) if not obj['Key'].endswith('/')]
    if not objects:raise ValueError('Empty reference prefix')
    for obj in objects:relative_key(obj['Key'],prefix)
    manifest=root/'download_manifest.json';previous={}
    if manifest.exists():previous={entry['key']:entry for entry in json.loads(manifest.read_text())['objects']}
    print(f'Reference surface: {len(objects)} objects, {sum(o["Size"] for o in objects):,} bytes',flush=True)
    def fetch(obj):
        relative=relative_key(obj['Key'],prefix);path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe local destination')
        old=previous.get(obj['Key'],{});sha=None
        if path.exists() and path.stat().st_size==obj['Size'] and old.get('etag')==obj['ETag']:
            sha=digest(path)
            if sha!=old.get('sha256'):sha=None
        if sha is None:
            temporary=path.with_name(path.name+'.download-part')
            if temporary.is_symlink():raise ValueError('Unsafe partial destination')
            client.download_file(source.netloc,obj['Key'],str(temporary),
                Config=TransferConfig(max_concurrency=2))
            if temporary.stat().st_size!=obj['Size']:raise ValueError('Downloaded size mismatch')
            sha=digest(temporary);temporary.replace(path)
        print(f'Reference downloaded {relative}: {obj["Size"]:,} bytes',flush=True)
        return dict(key=obj['Key'],relative=str(relative),bytes=obj['Size'],etag=obj['ETag'],
                    last_modified=obj['LastModified'].isoformat(),sha256=sha)
    with ThreadPoolExecutor(max_workers=cfg.get('workers',4)) as pool:entries=list(pool.map(fetch,objects))
    report=dict(source=cfg['source'],downloaded_utc=datetime.now(timezone.utc).isoformat(),
                object_count=len(entries),bytes=sum(e['bytes'] for e in entries),objects=entries)
    temp=manifest.with_suffix('.json.part');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(manifest)
    return report
