# Running several `infer.py` processes on one GPU

One `infer.py` process does not keep a GPU busy: tile reads and preprocessing leave the GPU
idle between batches. Running 2–4 processes on the same card, each on its own part of the
surface list, raises throughput.

Measured on one PHerc0800 wrap (2400 × 5820 output), with `--overlap 0.5`, Hann blending and
`--batch-size 1`, one wrap per configuration. Each concurrent output was compared with the single-process output of the
same card:

| GPU | 1 process (stock flags) | Recipe | Wraps/h | Output vs 1 process |
|---|---|---|---|---|
| RTX 3080 10 GB | 125 | 2 processes × `--num-workers 1`, `--gpu-normalize` | 242 | byte-identical |
| Radeon AI PRO R9700 32 GB | 54 | 4 processes, `--gpu-normalize` | 256 | max diff 1/255 |
| Instinct MI50 32 GB | 48 | 3–4 processes, stock flags | 125–133 | not byte-identical; max diff not recorded |

On the R9700 that is 256 / 54 = 4.7× the single-process rate.

Notes:

- `--gpu-normalize` (#2189) moves the per-tile normalization to the GPU. It matters most when
  several processes share the CPU.
- `--batch-size 8` with `--amp-dtype fp16` gave no gain on any of the three cards.
- Keep processes × `--num-workers` at or below the number of CPU cores. On Windows each worker
  loads its own copy of torch, so use 0 or 1 workers per process there.
- On AMD cards, MIOpen may pick different convolution algorithms in different processes, which
  is where the small differences there come from.

`--folder` runs every segment directory under one parent folder. Split the segments into one
parent folder per process with symlinks, then start one process per folder:

```bash
N=4
ls -d /path/to/segments/*/ | split -n r/$N - list_
i=0
for list in list_*; do
  mkdir -p part$i
  while read -r seg; do ln -s "$seg" part$i/; done < "$list"
  python infer.py --folder part$i --checkpoint-path model.ckpt --gpu-normalize --num-workers 1 &
  i=$((i + 1))
done
wait
```

Each output TIFF is written to `preds/` inside its segment directory, as in a single-process
run.

To check parity, run a few segments once with a single process and once inside the concurrent
run, each with its own `--output-prefix` (folder mode skips segments that already have a
prediction with the same name), then compare the two TIFFs with `cmp`.
