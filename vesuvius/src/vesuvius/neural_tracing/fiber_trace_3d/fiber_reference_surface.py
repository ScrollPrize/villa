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


def validate_refresh(root, source, prefix, objects):
    """Reject mixed snapshots before downloading; never remove user files."""
    manifest=root/'download_manifest.json'
    previous={}
    if manifest.exists():
        doc=json.loads(manifest.read_text())
        old=urlparse(doc['source']);new=urlparse(source)
        if (old.scheme,old.netloc,old.path.rstrip('/'))!=(new.scheme,new.netloc,new.path.rstrip('/')):
            raise ValueError('Reference source changed; use a new output_root')
        previous={entry['key']:entry for entry in doc['objects']}
    expected={relative_key(obj['Key'],prefix).as_posix() for obj in objects}
    metadata={'download_manifest.json','download_manifest.json.part','listing.json'}
    partials={name+'.download-part' for name in expected}
    if expected&(metadata|partials):
        raise ValueError('Reference objects conflict with download metadata or partial files')
    allowed=expected|metadata|partials
    stale=[]
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Unsafe local destination: '+str(path))
        if path.is_file() and path.relative_to(root).as_posix() not in allowed:
            stale.append(path.relative_to(root).as_posix())
    if stale:
        raise ValueError('Stale/unlisted reference files; use a clean output_root: '+', '.join(sorted(stale)))
    return previous


def run(cfg):
    from vc_reader.reader import ObjectStore
    source=urlparse(cfg['source'])
    if source.scheme!='s3' or not source.netloc:raise ValueError('Expected S3 source')
    prefix=source.path.lstrip('/').rstrip('/')+'/'
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    store=ObjectStore(f's3://{source.netloc}/{prefix}')
    listed=[dict(obj,Key=prefix+obj['key']) for obj in store.objects() if not obj['key'].endswith('/')]
    if cfg.get('list_only',False):
        report=dict(source=cfg['source'],objects=[dict(key=o['Key'],bytes=o['size']) for o in listed if '/' not in o['key']],
            prefixes=[prefix+name+'/' for name in store.children()])
        (root/'listing.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return report
    objects=[dict(Key=o['Key'],Size=o['size'],ETag=o['etag'],LastModified=o['last_modified']) for o in listed]
    if not objects:raise ValueError('Empty reference prefix')
    for obj in objects:relative_key(obj['Key'],prefix)
    manifest=root/'download_manifest.json'
    previous=validate_refresh(root,cfg['source'],prefix,objects)
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
            temporary.write_bytes(store.read(relative.as_posix()))
            if temporary.stat().st_size!=obj['Size']:raise ValueError('Downloaded size mismatch')
            sha=digest(temporary);temporary.replace(path)
        print(f'Reference downloaded {relative}: {obj["Size"]:,} bytes',flush=True)
        return dict(key=obj['Key'],relative=str(relative),bytes=obj['Size'],etag=obj['ETag'],
                    last_modified=obj['LastModified'],sha256=sha)
    with ThreadPoolExecutor(max_workers=cfg.get('workers',4)) as pool:entries=list(pool.map(fetch,objects))
    report=dict(source=cfg['source'],downloaded_utc=datetime.now(timezone.utc).isoformat(),
                object_count=len(entries),bytes=sum(e['bytes'] for e in entries),objects=entries)
    temp=manifest.with_suffix('.json.part');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(manifest)
    return report
