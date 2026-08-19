# Native 3D Trace2CP Benchmarks

Wall time is the shell `time` real value. CPU time is `user + sys`. Error rates
come from the final aggregate tracer output.

| Date       | Tracer | Git revision                                       | Fibers   | Snapshot                     | Settings            | Wall time   | CPU time    | err/kvx | err/m | Avg trace length |
|------------|--------|----------------------------------------------------|----------|------------------------------|---------------------|-------------|-------------|---------|-------|------------------|
| 2026-08-02 | Python | `07451e4eb0f7e610a43deaa94d51669ded65daca`         | `paul4`  | `s1a-128-sd2-single-25k`     | `python-sd2`        | 14m52.921s  | 51m04.529s  | 1.0     | 105.0 | 9.2 mm           |
| 2026-08-02 | Python | `07451e4eb0f7e610a43deaa94d51669ded65daca`         | `fiber1` | `s1a-128-sd2-single-25k`     | `python-sd2`        | 2m28.422s   | 10m48.048s  | 0.4     | 45.8  | 19.1 mm          |
| 2026-08-02 | Python | `e7e7a19c3b010c3d1185e1d6513e6e8feb235856`         | `paul4`  | `s1a-128-sd1-best91p5k`      | `python-sd1-step16` | 43m30.494s  | 56m23.290s  | 0.3     | 54.9  | 17.0 mm          |
| 2026-08-19 | Python | `9095ba351b6fea1c37253190cedddab5c97373f6`         | `paul4`  | `s1a-128-s0-single-140p7k`   | `python-s0-sd2-step8` | 131m44.174s | 75m56.251s | 0.2     | 71.6  | 13.2 mm          |
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
| `python-s0-sd2-step32` | `9095ba351b6fea1c37253190cedddab5c97373f6`        | `metric_s0_s1_single.json`              | `--inference-scaledown-power 2 --step-voxels 32`     |
