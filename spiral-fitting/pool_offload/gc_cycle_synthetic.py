"""Synthetic reproduction of the fit's 'device memory grows with the iteration count' (GC_CYCLE_LEAK.md) without the fit, its store or its data.
Usage: python gc_cycle_synthetic.py DEVICE HEAP_OBJECTS STEPS TENSOR_MiB POLICIES [RESULT.json]
  e.g. python gc_cycle_synthetic.py cuda 2e7 40 64 default,freeze,every_step     (DEVICE is cuda or cpu)
Mechanism under test: per-step objects that sit in REFERENCE CYCLES, each holding a big tensor, and that live long enough to be PROMOTED to the oldest generation before they die; Python frees
them only at an oldest-generation collection, which is triggered by allocation counts (a collection of the oldest generation runs only when the objects that survived the younger ones have grown by 25 % of what the
last full collection kept), not by device memory.  A large permanent heap (the fit's 50.8 M objects after startup) therefore makes full collections RARE.
Setup: build a heap of N small objects (a list of lists), then for S steps: create a cycle object holding a T-MiB tensor, churn through ~200 k short-lived container allocations (so the cycle is promoted),
drop the cycle.  Policies: default (Python's collector), freeze (gc.freeze() once after the heap is built, then gc.collect() every step: the fitter's default), every-step (gc.collect() with no freeze), off-after-freeze.
Reported: peak tensor memory (CUDA max_memory_allocated, or process RSS growth on CPU), full-collection count and total collector seconds, steps per second."""
import gc, sys, time, json, os, resource, torch
dev = sys.argv[1]; N = int(float(sys.argv[2])); S = int(sys.argv[3]); T = int(sys.argv[4]); policies = sys.argv[5].split(',')
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
res = {}
if dev == 'cpu' and len(policies) > 1:
    # Process RSS high-water marks cannot be reset, so on the CPU every policy runs in its own fresh process. A failed child fails the run (non-zero exit, no silent gap), and the
    # optional RESULT.json collects every child's result.
    import subprocess, tempfile
    failed = []
    for pol in policies:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp: child_json = tmp.name
        run = subprocess.run([sys.executable, __file__, dev, sys.argv[2], sys.argv[3], sys.argv[4], pol, child_json], capture_output=True, text=True)
        print(run.stdout.strip(), flush=True)
        if run.returncode != 0:
            failed.append(pol); print(f"policy {pol} FAILED (exit {run.returncode}): {run.stderr.strip()[-400:]}", file=sys.stderr, flush=True)
        else:
            try: res.update(json.load(open(child_json)))
            except Exception as error: failed.append(pol); print(f"policy {pol}: no result file ({error})", file=sys.stderr, flush=True)
        try: os.unlink(child_json)
        except OSError: pass
    if len(sys.argv) > 6: json.dump(res, open(sys.argv[6], 'w'), indent=1)
    sys.exit(1 if failed else 0)
for pol in policies:
    gc.enable(); gc.unfreeze(); gc.collect(); heap = [[i] for i in range(N)]; gc.collect()
    if dev == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base_rss = rss(); stats = {'gen2': 0, 'sec': 0.0}; t0 = [0.0]
    def cb(phase, info):
        if info.get('generation') == 2:
            if phase == 'start': t0[0] = time.perf_counter()
            else: stats['gen2'] += 1; stats['sec'] += time.perf_counter() - t0[0]
    gc.callbacks.append(cb)
    if pol in ('freeze', 'off_after_freeze'): gc.collect(); gc.freeze()
    t = time.time(); live = 0
    for s in range(S):
        obj = {}; obj['t'] = (torch.zeros if dev == 'cpu' else torch.empty)(T * 262144, dtype=torch.float32, device=dev); obj['self'] = obj           # a reference cycle holding T MiB
        churn = [{'k': j} for j in range(200000)]; del churn                                                      # allocation churn: promotes the cycle to the oldest generation
        del obj
        if pol in ('freeze', 'every_step'): gc.collect()
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == 'cuda' else (rss() - base_rss) * 1024
    res[pol] = dict(peak_tensor_MiB=round(peak), full_collections=stats['gen2'], collector_seconds=round(stats['sec'], 2), seconds_per_step=round((time.time() - t) / S, 3), frozen_objects_at_end=gc.get_freeze_count())
    print(dev, 'heap', N, 'steps', S, 'tensor MiB', T, pol, res[pol], flush=True)
    gc.callbacks.remove(cb); gc.unfreeze(); del heap; gc.collect()
if len(sys.argv) > 6: json.dump(res, open(sys.argv[6], 'w'), indent=1)
