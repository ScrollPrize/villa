"""DataLoader throughput for a run's args (no model):
  python scripts/loader_bench.py WORKERS SECONDS -- <train.py crop/sample args>"""
import os, sys, time
import numpy as np, torch
if __name__ == "__main__":
    from vesuvius.neural_tracing.fiber_follow import train as T
    nw, secs = int(sys.argv[1]), float(sys.argv[2])
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap_args = ["--fiber-zarrs", "/home/sean/Documents/volpkgs/s1_2um.volpkg/20260411134726-fibers-20260915212757-L1_masked",
               "--fibers", "/mnt/bigpc/spiral_dataset_working/fibers", "--name", "_bench"] + argv
    from vesuvius.neural_tracing.fiber_follow.data import (FollowDataset, OnPolicyStates, SampleConfig, ZBand,
                                                           gt_presence, load_fibers, mark_breaks, split_fibers)
    from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
    from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec
    import argparse
    p = argparse.ArgumentParser(); [p.add_argument(a) for a in ()]
    kv = dict(zip(argv[::2], argv[1::2]))
    crop = CropSpec(depth=int(kv.get("--crop-depth", 64)), width=int(kv.get("--crop-width", 64)),
                    behind=int(kv.get("--crop-behind", 16)), gate_direction=True)
    cfg = SampleConfig(crop=crop, n_history=int(kv.get("--n-history", 128)), n_future=16,
                       lateral_sigmas=(0.5, 1.2, 2.5), angle_sigmas_deg=(5, 12, 25), history_wobble=1.0)
    spec = FiberVolumeSpec(ap_args[1])
    fibers = load_fibers(ap_args[3]); band = ZBand(45000 / 8, 48500 / 8)
    FF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mark_breaks(fibers, gt_presence(fibers, FiberVolume(spec), os.path.join(FF, "output", "gt_presence_controlled_v2.npz")))
    tr, _ = split_fibers(fibers, band)
    op = OnPolicyStates.load(kv["--onpolicy"]) if "--onpolicy" in kv else None
    ds = FollowDataset(tr, spec, cfg, band, cache_bytes=1 << 30, onpolicy=[op] if op else [], onpolicy_prob=0.3, hard_prob=0.2)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=nw, **({"prefetch_factor": 2} if nw else {}))
    it = iter(dl)
    for _ in range(nw * 6): next(it)
    t = time.time(); n = 0
    while time.time() - t < secs:
        n += len(next(it)["x"])
    print(f"workers {nw} fused={os.environ.get('FIBER_FOLLOW_FUSED', '1')} crop {crop.depth}x{crop.width}: {n / (time.time() - t):.0f} samples/s", flush=True)
    if nw: it._shutdown_workers()
    del it, dl
