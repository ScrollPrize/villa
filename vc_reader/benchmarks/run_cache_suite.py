#!/usr/bin/env python3
"""Launch and report the C++ cache workload matrix; no cache logic lives here."""
import argparse
import csv
import hashlib
import io
import json
import platform
import resource
import shutil
import time
import os
import sys
import statistics
import subprocess
from pathlib import Path

CASES = {
    "hits": "Uniform cached reads",
    "hot": "Reads contending on 16 keys",
    "miss": "Absent-key reads",
    "insert": "65,536 distinct inserts (finite workload)",
    "replace": "Replace existing cached values",
    "evict": "Insert/evict, retained payloads",
    "allocate_evict": "Insert/evict with fresh allocations",
    "mixed": "80% hot reads, 20% writes; misses not refilled",
    "mixed_shrink": "Mixed traffic plus budget changes every 10 ms",
    "burst": "Write fraction 20% → 90% → 20%",
    "paced": "Mixed traffic paced at 1 M operations/s",
    "adaptive": "Burst traffic paced at 250 k operations/s",
}


def report(root, results, workers):
    virtual = any('GiB' in r['policy'] for r in results)
    historical = json.loads((root/'metadata.json').read_text()).get('backend') == 'pre-reader'
    lines = ["# Complete synthetic cache suite", "",
             "Median of three repetitions by default; M operations/s (reads and puts, not bytes). "
             "Raw repetitions, exact commands, latencies and accounting are in results.json. "
             "The finite insertion case is not a steady-state throughput test. "
             "Optional post-workload headroom replenishment is excluded.", ""]
    if virtual:
        capacities = sorted({float(r['policy'].split()[0]) for r in results})
        lines = ['# Full virtual-capacity cache suite', '',
                 'Foreground-worker ablation with fixed shard count; four maintenance workers, eight-tail sampling. Six consecutive 2-second windows per ordinary case, not independent repetitions. Table entries are median M requests/s of the last three windows. Insert is the finite parallel prefill (synchronous, before maintenance starts); isolated shrink is entries reclaimed/s for one 5% reduction. Mixed reads refill misses. Allocation/eviction immediately replaces each fresh 4 KiB payload with a shared dummy: two puts per request, bounded live payloads. No final headroom drain.', '',
                 '| Case | Workers | ' + ' | '.join(f'{size:g} GiB' for size in capacities) + ' |',
                 '|---|---:|' + '---:|' * len(capacities)]
        for case in [*CASES,'shrink_only']:
            for w in workers:
                cells=[]
                for size in capacities:
                    policy=f'{size:g} GiB '+('pre-reader' if historical else 'sampled8')
                    row=next((r for r in results if (r['case'],r['workers'],r['policy'])==(case,w,policy)),None)
                    cells.append('—' if row is None else 'ERROR' if 'error' in row else f"{row['last_three_median']/1e6:.2f}")
                lines.append(f'| {case} | {w} | '+' | '.join(cells)+' |')
        lines.append('')
        if historical:
            lines[2] = lines[2].replace('Foreground-worker ablation with fixed shard count; four maintenance workers, eight-tail sampling.',
                'Historical VC ChunkCache decoded-storage benchmark: one source, synchronous shared-byte-budget enforcement, no background maintenance. Original implementation compiled from the checkout recorded in metadata; only benchmark access is added. Numeric source keys and opaque shared payloads avoid I/O, decode and scheduler overhead. Unsupported maintenance counters are not measured.')
        lines += ['## Physical memory', '', '| Fixture | Prefill seconds | Peak RSS GiB | Peak virtual GiB | AS limit GiB | RSS guard GiB |', '|---|---:|---:|---:|---:|---:|']
        for path in sorted(root.glob('virtual_*.stderr'),key=lambda p:tuple(map(int,p.stem.split('_')[1:]))):
            peak=0;prefill='—'
            for line in path.read_text().splitlines():
                fields=line.split(',')
                if fields[0] in ('prefill_peak_rss_bytes','peak_rss_bytes'):
                    peak=max(peak,int(fields[-1]))
                elif fields[0]=='prefill_complete':
                    prefill=f'{float(fields[-1]):.2f}'
            limits_path=path.with_suffix('.limits.json')
            limits=json.loads(limits_path.read_text()) if limits_path.exists() else {}
            peak=max(peak,limits.get('peak_rss_bytes',0))
            row=next((r for r in results if path.stem==f"virtual_{r['command'][1]}_{r['workers']}"),{})
            address=limits.get('address_limit_gib',row.get('address_limit_gib','—'))
            rss_guard=limits.get('rss_limit_gib',row.get('rss_limit_gib',0)) or '—'
            virtual_peak=f"{limits['peak_address_bytes']/2**30:.2f}" if limits.get('peak_address_bytes') else '—'
            lines.append(f'| {path.stem} | {prefill} | {peak/2**30:.2f} | {virtual_peak} | {address} | {rss_guard} |')
        lines.append('')
    for policy in dict.fromkeys(r['policy'] for r in results):
        lines += [f"## {policy}", "", "| Case | " + " | ".join(map(str, workers)) + " |",
                  "|---|" + "---:|" * len(workers)]
        for case in [*CASES,'shrink_only'] if virtual else CASES:
            cells = []
            for worker in workers:
                row = next((r for r in results if (r['case'], r['workers'], r['policy']) == (case, worker, policy)), None)
                cells.append("—" if row is None else "ERROR" if 'error' in row else f"{row['last_three_median' if virtual else 'median']/1e6:.2f}")
            lines.append(f"| {case} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Workloads", "", "| Case | Meaning |", "|---|---|"]
    lines += [f"| {case} | {meaning} |" for case, meaning in CASES.items()] if not virtual else [
        '| mixed variants | 80% reads with refills; mixed_shrink performs one 5% budget reduction halfway through each window |',
        '| insert | Full-capacity prefill, one finite measurement, maintenance off |',
        '| replace | Random initial keys; maintenance may have evicted a small fraction, so some operations reinsert |']
    lines += ["| shrink_only | One 5% resident-budget reduction using the selected maintenance policy; entries evicted/s |" if virtual else "| shrink_only | Single-thread synchronous budget shrink, 32 cycles; entries evicted/s |", "",
              "## Detailed measurements", "",
              "| Policy | Case | Workers | Mean / median / min / max M/s | Per worker k/s | Read hit % | Put p50 / p95 / p99 µs | Assist puts % | Evicted entries/batch | Peak sampled excess bytes |",
              "|---|---|---:|---|---:|---:|---|---:|---:|---:|"]
    for r in results:
        if 'error' in r:
            lines.append(f"| {r['policy']} | {r['case']} | {r['workers']} | ERROR: {r['error'].replace(chr(10), ' ')} | | | | | | |")
            continue
        rows = r['rows']
        def median(key):
            return statistics.median(float(x.get(key) or 0) for x in rows)
        def ratio(a, b):
            return statistics.median(float(x.get(a) or 0)/max(1,float(x.get(b) or 0)) for x in rows)
        rates = ' / '.join(f"{r[k]/1e6:.2f}" for k in ('mean','median','min','max'))
        has_puts=any(float(x.get('puts') or 0) for x in rows)
        latencies = ' / '.join(f"{median(k)/1000:.1f}" for k in ('put_p50_ns','put_p95_ns','put_p99_ns')) if has_puts else '—'
        hits=f"{100*ratio('hits','reads'):.2f}" if any(float(x.get('reads') or 0) for x in rows) else '—'
        assist=f"{100*ratio('assist_puts','puts'):.2f}" if has_puts and not historical else '—'
        batches=f"{ratio('evicted_entries','batches'):.1f}" if not historical else '—'
        per_worker=f"{r['median']/r['workers']/1000:.1f}" if r['case']!='shrink_only' else '—'
        lines.append(f"| {r['policy']} | {r['case']} | {r['workers']} | {rates} | {per_worker} | {hits} | {latencies} | {assist} | {batches} | {max(float(x.get('peak_sampled_overshoot') or 0) for x in rows):.0f} |")
    if virtual:
        lines += ['', '## Window progression', '', '| Capacity | Case | Workers | Consecutive M ops/s |', '|---|---|---:|---|']
        for r in results:
            if 'error' not in r:
                lines.append(f"| {r['policy']} | {r['case']} | {r['workers']} | "+' / '.join(f"{float(x['chunks_per_second'])/1e6:.2f}" for x in r['rows'])+' |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def bounded_run(command, out, err, args, metrics_path):
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        limit=int(args.memory_limit_gib*2**30)
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    if args.rss_limit_gib and sys.platform!='linux':
        raise SystemExit('RSS monitoring requires Linux; choose --rss-limit-gib 0 to use only RLIMIT_AS')
    env=dict(os.environ,VCR_BENCH_PREFILL_SECONDS=str(args.prefill_seconds))
    stats=dict(address_limit_gib=args.memory_limit_gib,rss_limit_gib=args.rss_limit_gib,
               peak_rss_bytes=0,peak_address_bytes=0)
    error=None;start=time.monotonic()
    with subprocess.Popen(command,stdout=out,stderr=err,preexec_fn=limits,env=env) as child:
        try:
            while child.poll() is None:
                if sys.platform=='linux':
                    try:
                        fields={line.split(':')[0]:line.split(':')[1].split()[0]
                                for line in Path(f'/proc/{child.pid}/status').read_text().splitlines()
                                if line.startswith(('VmRSS:','VmPeak:'))}
                        rss=int(fields.get('VmRSS',0))*1024
                        stats['peak_rss_bytes']=max(stats['peak_rss_bytes'],rss)
                        stats['peak_address_bytes']=max(stats['peak_address_bytes'],int(fields.get('VmPeak',0))*1024)
                        if args.rss_limit_gib and rss>args.rss_limit_gib*2**30:
                            error='RSS safety limit exceeded'
                    except FileNotFoundError:
                        pass
                if time.monotonic()-start>args.prefill_seconds+180:
                    error='Total setup plus measurement timeout'
                if error:
                    child.kill();break
                time.sleep(.25)
            code=child.wait()
            if code and not error:error=f'exit {code}'
        finally:
            if child.poll() is None:child.kill();child.wait()
            stats.update(seconds=time.monotonic()-start,error=error)
            metrics_path.write_text(json.dumps(stats,indent=2)+'\n')
    return error


def virtual_suite(args, binary, root, cores):
    """Same report format, using the real-entry-count C++ fixture."""
    workers = args.workers or sorted(set([1,8,32,64,cores,2*cores,10*cores]))
    results = []
    reusable=[]
    failed=None
    if args.only_failed_from:
        previous=json.loads((args.only_failed_from/'results.json').read_text())
        failed={(r['policy'],r['workers']) for r in previous if 'error' in r or 'process_error' in r}
    if args.reuse_from:
        previous=json.loads((args.reuse_from/'metadata.json').read_text())
        current=json.loads((root/'metadata.json').read_text())
        if previous.get('backend','sampled8')!=current.get('backend','sampled8'):
            raise SystemExit('Cannot reuse a different cache backend')
        for field in ('sha256','host','shards','payload_bytes','seconds','windows',
                      'maintenance_workers','candidate_scan','memory_limit_gib','allocation_case'):
            if previous.get(field)!=current.get(field):
                raise SystemExit(f'Cannot reuse results: {field} differs')
        reusable=json.loads((args.reuse_from/'results.json').read_text())
    names = {'ops_per_second':'chunks_per_second', 'p50_ns':'put_p50_ns',
             'p95_ns':'put_p95_ns', 'p99_ns':'put_p99_ns', 'scans':'selection_scans'}
    for capacity in args.capacities:
        entries = int(capacity * 2**30 // 4096)
        for w in workers:
            backend='pre-reader' if args.backend=='pre-reader' else 'sampled8'
            policy=f'{entries*4096/2**30:g} GiB {backend}'
            if failed is not None and (policy,w) not in failed:continue
            command=[str(binary),str(entries),str(w),str(1 if args.backend=='pre-reader' else cores),
                     '0' if args.backend=='pre-reader' else '4','2',
                     '0' if args.backend=='pre-reader' else '8','0','6','full']
            name=f'virtual_{entries}_{w}'
            reused=[r for r in reusable if r['workers']==w and r['policy']==policy]
            if len(reused)==len(CASES)+1 and not any('error' in r or 'process_error' in r for r in reused):
                for suffix in ('.csv','.stderr'):
                    shutil.copyfile(args.reuse_from/(name+suffix),root/(name+suffix))
                results.extend(dict(r,reused_from=str(args.reuse_from)) for r in reused)
                (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                report(root,results,workers)
                print('Reused validated '+name,flush=True)
                continue
            print('Starting '+name,flush=True)
            with (root/(name+'.csv')).open('w') as out, (root/(name+'.stderr')).open('w') as err:
                error=bounded_run(command,out,err,args,root/(name+'.limits.json'))
            raw=list(csv.DictReader(io.StringIO((root/(name+'.csv')).read_text())))
            for case in [*CASES,'shrink_only']:
                rows=[{names.get(k,k):v for k,v in r.items()} for r in raw if r['mode']==case]
                # Finite insertion is the full-entry-count prefill. Shrink waits
                # only for the requested smaller budget, never future headroom.
                prefix={'insert':'insert_result,','shrink_only':'shrink_result,'}.get(case)
                if prefix:
                    for line in (root/(name+'.stderr')).read_text().splitlines():
                        if line.startswith(prefix):
                            fields=line.split(',')
                            rows=[dict(chunks_per_second=fields[1],seconds=fields[2])]
                item=dict(case=case,workers=w,policy=policy,command=command,rows=rows,
                          binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                          address_limit_gib=args.memory_limit_gib,rss_limit_gib=args.rss_limit_gib,
                          prefill_seconds=args.prefill_seconds)
                if rows:
                    values=[float(r['chunks_per_second']) for r in rows]
                    item.update(mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values))
                    item['last_three_median']=statistics.median(values[-3:])
                else:
                    item['error']=error or 'Missing workload output'
                if error:item['process_error']=error
                results.append(item)
            (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            report(root,results,workers)
            print(f'Finished {name}: {error or "OK"}',flush=True)
    if any('error' in r or 'process_error' in r for r in results):
        raise SystemExit('Some combinations failed; all results and failures preserved')


def combine_reports(args):
    """Combine disjoint successes and explicit retries, retaining source metadata."""
    combined={};sources={};settings=None;workers=set()
    fields=('shards','payload_bytes','seconds','windows','maintenance_workers','candidate_scan','allocation_case')
    for source in args.combine_from:
        meta=json.loads((source/'metadata.json').read_text())
        selected={k:meta.get(k) for k in fields}
        selected['backend']=meta.get('backend','sampled8')
        if settings is not None and settings!=selected:
            raise SystemExit('Measured workload settings differ; refusing to combine')
        settings=selected;sources[str(source)]=meta;workers.update(meta['workers'])
        for row in json.loads((source/'results.json').read_text()):
            key=(row['policy'],row['workers'],row['case'])
            old=combined.get(key)
            if old and 'error' not in old and 'process_error' not in old:
                raise SystemExit(f'Refusing to overwrite an already successful result: {key}')
            combined[key]=dict(row,source_directory=str(source),binary_sha256=meta['sha256'],
                               address_limit_gib=meta['memory_limit_gib'],rss_limit_gib=meta.get('rss_limit_gib',0))
    args.output.mkdir(parents=True,exist_ok=False)
    for source in args.combine_from:
        for path in source.glob('virtual_*'):
            if path.is_file():shutil.copyfile(path,args.output/path.name)
    rows=list(combined.values())
    (args.output/'metadata.json').write_text(json.dumps(dict(workers=sorted(workers),sources=sources,comparison_settings=settings),indent=2)+'\n')
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report(args.output,rows,sorted(workers))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, nargs='+')
    parser.add_argument('--seconds', type=float, default=.5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--virtual', action='store_true', help='Full suite at .25/64/512 GiB with bench_cache_scale; six consecutive 2s windows')
    parser.add_argument('--capacities', type=float, nargs='+', choices=(.25, 64, 512), default=(.25, 64, 512),
                        help='Virtual cache capacities in GiB')
    parser.add_argument('--backend',choices=('sampled8','pre-reader'),default='sampled8')
    parser.add_argument('--historical-source',type=Path,help='Historical checkout used to build the pre-reader executable')
    parser.add_argument('--report-only', action='store_true', help='Regenerate tables from existing results without running benchmarks')
    parser.add_argument('--memory-limit-gib', type=float, default=256, help='Virtual-suite child address-space limit (GiB), enforced with RLIMIT_AS')
    parser.add_argument('--reuse-from',type=Path,help='Reuse completed virtual combinations only when executable hash and benchmark settings match')
    parser.add_argument('--only-failed-from',type=Path,help='Run only combinations that failed in this previous results directory')
    parser.add_argument('--rss-limit-gib',type=float,default=64 if sys.platform=='linux' else 0,help='Linux RSS watchdog ceiling, sampled every 250 ms; 0 disables')
    parser.add_argument('--prefill-seconds',type=float,default=300,help='Fixture setup guard; does not change measured window duration')
    parser.add_argument('--combine-from',type=Path,nargs='+',help='Combine disjoint successful runs and retries into a new report; retain source metadata')
    args = parser.parse_args()
    if args.combine_from:
        combine_reports(args);return
    if args.report_only:
        metadata=json.loads((args.output/'metadata.json').read_text())
        report(args.output,json.loads((args.output/'results.json').read_text()),metadata['workers'])
        return
    binary = args.binary.resolve()
    import os
    cores = len(os.sched_getaffinity(0)) if args.virtual and hasattr(os,'sched_getaffinity') else (os.cpu_count() if args.virtual else int(subprocess.check_output([str(binary), 'cores'], text=True)))
    workers = args.workers or sorted(set([1,8,32,64,cores,2*cores,10*cores]))
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    metadata = dict(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    host=platform.node(), cores=cores, workers=workers, shards=cores,
                    repeats=args.repeats, seconds=args.seconds, payload_bytes=256,
                    pressure_entries=4096, key_count=65536, maintenance_workers=4,
                    candidate_scan=8, build_type='Use Release; verify CMakeCache.txt')
    metadata['backend']=args.backend
    if args.backend=='pre-reader':
        if not args.virtual or not args.historical_source:
            raise SystemExit('pre-reader requires --virtual and --historical-source')
        metadata.update(shards=1,maintenance_workers=0,candidate_scan=0,
                        historical_source=str(args.historical_source.resolve()),
                        historical_commit=subprocess.check_output(['git','-C',str(args.historical_source),'rev-parse','HEAD'],text=True).strip())
        tracked=subprocess.check_output(['git','-C',str(args.historical_source),'status','--porcelain','--untracked-files=no'],text=True)
        if tracked.strip():raise SystemExit('Historical source has tracked modifications')
    (root/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.virtual:
        metadata.update(workers=workers,payload_bytes=4096,capacity_entries=[int(capacity * 2**30 // 4096) for capacity in args.capacities],seconds=2,windows=6,repeats=None,pressure_entries=None,key_count=None,memory_limit_gib=args.memory_limit_gib,allocation_case='Fresh payload followed by shared replacement; two puts per request',reuse_from=str(args.reuse_from) if args.reuse_from else None,rss_limit_gib=args.rss_limit_gib,prefill_seconds=args.prefill_seconds)
        (root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        virtual_suite(args,binary,root,cores)
        return
    results = []
    jobs = [('shrink_only',1,'sync',0)]
    # Interleave the two policies to reduce long-run host-load bias.
    jobs += [(case,w,policy,m) for w in workers for case in CASES
             for policy,m in [('sync',0),('sampled8',4)]]
    for index,(case,w,policy,maintenance) in enumerate(jobs):
        mode = {'paced':'mixed','adaptive':'burst'}.get(case,case)
        seconds = max(args.seconds, 2 if case=='adaptive' else 1 if case in ('paced','burst') else args.seconds)
        rate = 1000000 if case=='paced' else 250000 if case=='adaptive' else 0
        command = [str(binary),str(w),str(args.repeats),str(seconds),mode,str(cores),'0',str(maintenance),str(rate),'8']
        result = dict(case=case,workers=w,policy=policy,command=command)
        name = f'{case}_{w}_{policy}'
        try:
            run = subprocess.run(command, text=True, capture_output=True, timeout=28)
            (root/(name+'.csv')).write_text(run.stdout)
            (root/(name+'.stderr')).write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(run.stderr[-500:] or f'exit {run.returncode}')
            rows = list(csv.DictReader(io.StringIO(run.stdout)))
            if len(rows) != args.repeats:
                raise RuntimeError(f'Expected {args.repeats} repetitions, got {len(rows)}')
            values = [float(r['chunks_per_second']) for r in rows]
            result.update(rows=rows, mean=statistics.mean(values), median=statistics.median(values), min=min(values), max=max(values))
        except (subprocess.TimeoutExpired, RuntimeError) as error:
            result['error'] = str(error)
        results.append(result)
        (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        report(root,results,workers)
        print(f"{index+1}/{len(jobs)} {name}: " + (result['error'] if 'error' in result else f"{result['median']/1e6:.2f} M/s"),flush=True)
    if any('error' in r for r in results):
        raise SystemExit('Some cases failed; retained explicitly in report.md')


if __name__ == '__main__':
    main()
