# Native 3D Trace2CP Benchmarks

Wall time is the shell `time` real value. CPU time is `user + sys`. Error rates
come from the final aggregate tracer output.

| Date       | Tracer | Git revision                                       | Fibers   | Snapshot                     | Settings            | Wall time   | CPU time    | err/kvx | err/m | Avg trace length |
|------------|--------|----------------------------------------------------|----------|------------------------------|---------------------|-------------|-------------|---------|-------|------------------|
| 2026-08-02 | Python | `07451e4eb0f7e610a43deaa94d51669ded65daca`         | `paul4`  | `s1a-128-sd2-single-25k`     | `python-sd2`        | 14m52.921s  | 51m04.529s  | 1.0     | 105.0 | 9.2 mm           |
| 2026-08-02 | Python | `07451e4eb0f7e610a43deaa94d51669ded65daca`         | `fiber1` | `s1a-128-sd2-single-25k`     | `python-sd2`        | 2m28.422s   | 10m48.048s  | 0.4     | 45.8  | 19.1 mm          |
| 2026-08-02 | Python | `e7e7a19c3b010c3d1185e1d6513e6e8feb235856`         | `paul4`  | `s1a-128-sd1-best91p5k`      | `python-sd1-step16` | 43m30.494s  | 56m23.290s  | 0.3     | 54.9  | 17.0 mm          |
| 2026-08-19 | Python | `9095ba351b6fea1c37253190cedddab5c97373f6`         | `paul4`  | `s1a-128-s0-single-140p7k`   | `python-s0-sd2-step8` | 131m44.174s | 75m56.251s | 0.2     | 71.6  | 13.2 mm          |
| 2026-08-19 | Python | `173280dc9659318da606034ee90246acf8c4015c`         | `paul4`  | `s1a-128-s0-single-140p7k`   | `python-s0-sd2-step16` | 53m42.996s | 76m53.541s | 0.2     | 72.6  | 13.1 mm          |
| 2026-08-19 | Python | `9095ba351b6fea1c37253190cedddab5c97373f6`         | `paul4`  | `s1a-128-s0-single-140p7k`   | `python-s0-sd2-step32` | 136m11.426s | 97m13.624s | 0.2     | 67.7  | 14.0 mm          |

## Fibers

Directory: `$VES/data/train_fibers/fibers_test_paul_4`

| Fibers   | Filename                                     |
|----------|----------------------------------------------|
| `paul4`  | `kb_20260605T150824406_000001.json`          |
| `paul4`  | `kb_20260606T024249299_000015.json`          |
| `paul4`  | `kb_20260607T023637101_000020.json`          |
| `paul4`  | `kb_20260623T152213524_000129.json`          |
| `fiber1` | `kb_20260605T150824406_000001.json`          |

## Snapshots

| Snapshot                     | Checkpoint                                                        |
|------------------------------|-------------------------------------------------------------------|
| `s1a-128-sd2-single-25k`     | `s1a_128_2_single_8x8_20260727_161616/best_25_9k.pt`              |
| `s1a-128-sd1-best91p5k`      | `s1a_128_1_single_8x8_20260801_084232/best91_5k.pt`               |
| `s1a-128-s0-single-140p7k`   | `s1a_128_0_single_8x8_20260809_194545/snapshots/best_140_7k.pt`   |

## Settings

| Settings            | Defaults revision                                   | Input selection                         | Non-default tracer arguments                         |
|---------------------|-----------------------------------------------------|-----------------------------------------|------------------------------------------------------|
| `python-sd2`        | `07451e4eb0f7e610a43deaa94d51669ded65daca`          | `metric_sd2_s1_single.json`             | `--inference-scaledown-power 2`                      |
| `python-sd1-step16` | `e7e7a19c3b010c3d1185e1d6513e6e8feb235856`          | config-free OME-Zarr group `/1`         | `--inference-scaledown-power 2 --step-voxels 16`     |
| `python-s0-sd2-step8` | `9095ba351b6fea1c37253190cedddab5c97373f6`         | `metric_s0_s1_single.json`              | `--inference-scaledown-power 2 --step-voxels 8`      |
| `python-s0-sd2-step16` | `173280dc9659318da606034ee90246acf8c4015c`        | `metric_s0_s1_single.json`              | `--inference-scaledown-power 2 --step-voxels 16`     |
| `python-s0-sd2-step32` | `9095ba351b6fea1c37253190cedddab5c97373f6`        | `metric_s0_s1_single.json`              | `--inference-scaledown-power 2 --step-voxels 32`     |

## Run Details

### `python-s0-sd2-step16`

Command:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/metric_s0_s1_single.json --checkpoint $VES/data/fiber/snapshots/s1a_128_0_single_8x8_20260809_194545/snapshots/best_140_7k.pt --export-dir ./ --fiber-json $VES/data/train_fibers/fibers_test_paul_4/kb_202606*.json --beam-lookahead-steps 2 --beam-width 8 --smoothness-normal-weight 0.1 --smoothness-tangent-weight 10.0 --core-margin-voxels 48 --inference-patch-shape-zyx 128 128 128 --inference-scaledown-power 2 --step-voxels 16
```

- First fiber: 60 restarts over 266 segments, `err/kvx=0.3`,
  `err/m=118.0`, average trace length 8.3 mm.
- First-fiber trace time: 1,670.944 s wall, 1,412.020 s CPU.
- Inference cache: 17,503 inferred blocks, zero evictions, 0.345 GiB.
- Four-fiber aggregate: 74 restarts over 435 segments, `err/kvx=0.2`,
  `err/m=72.6`, average trace length 13.1 mm.
- Shell time: 53m42.996s wall, 73m51.368s user, 3m02.173s system.
