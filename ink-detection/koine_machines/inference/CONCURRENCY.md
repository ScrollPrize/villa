# Running several `infer.py` processes on one GPU

One `infer.py` process does not keep a GPU busy: tile reads and preprocessing leave the GPU
idle between batches. Running 2–4 processes on the same card, each on its own part of the
surface list, raises throughput.

Measured on six Reader v3 windings (PHerc0800, 0125, 0211 and 0268 surface volumes) with
`--stride 192 --amp-dtype fp16 --no-compile` (MIOpen off on the AMD cards). The windings are split round-robin
over the processes, and the time runs until every process is done. Each value is the median
of 3 interleaved rounds after a warm-up, with nothing else running on the machine:

| GPU | 1 process | 2 processes | 4 processes | Output vs 1 process |
|---|---|---|---|---|
| RTX 3080 10 GB | 40.8 s | 33.1 s (1.23×) | 34.4 s (1.19×) | byte-identical |
| Radeon AI PRO R9700 32 GB | 74.7 s | 64.4 s (1.16×) | 51.9 s (1.44×) | byte-identical |
| Instinct MI50 32 GB | 102.2 s | 81.0 s (1.26×) | 77.7 s (1.32×) | byte-identical |

With `--gpu-normalize` the same runs were within 3% of these times on the R9700 and MI50,
and 1–4% slower on the RTX 3080. On the RTX 3080, 2 processes is the best setting.

Notes:

- `--gpu-normalize` (#2189) moves the per-tile normalization to the GPU. It made no measurable
  difference in the runs above, where the CPU was not the limit.
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
