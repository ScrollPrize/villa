# PHercParis4 3D ink: Gambiarra

Train and run `ink_paris4_3d_4.8um_gambiarra`, a residual 3D U-Net for PHercParis4 ink detection at 4.8 µm. The released checkpoint has ~24.5M parameters and comes from student update 25,000.

The released model came from four DINO teacher stages and two student stages. The package uses Villa's `dinovol_2` backbone and `vesuvius` network implementation.

## Weights and scribbles

| Artifact | Location | Files |
|---|---|---|
| Student weights and model card | [`scrollprize/ink_paris4_3d_4.8um_gambiarra`](https://huggingface.co/scrollprize/ink_paris4_3d_4.8um_gambiarra) | `model.safetensors`, `config.json`, `README.md`, `RECIPE.md`, `LICENSE` |
| Labeled scribbles | Prefix `ink_paris4_3d_4.8um_gambiarra/` in the [ScrollPrize datasets bucket](https://huggingface.co/buckets/scrollprize/datasets/) | `paris4_ink_scribbles.vc3d.json`, `README.md`, `LICENSE` |

Download [model.safetensors](https://huggingface.co/scrollprize/ink_paris4_3d_4.8um_gambiarra/resolve/main/model.safetensors) and pass its local path as `--weights`. Training downloads the scribbles from Hugging Face on first use and reuses the cached file. `assets/unlabelled_pool.csv` lists the CT parents for unlabelled student views.

## Environment and inputs

Use the `ink-paris4-3d-gambiarra` Villa branch, Python 3.14, and a CUDA GPU:

```bash
VILLA=/path/to/villa
cd "$VILLA/vesuvius"
uv sync --frozen --extra models --no-install-package volume-cartographer
source .venv/bin/activate
export PYTHONPATH="$VILLA/dinovol${PYTHONPATH:+:$PYTHONPATH}"
P=vesuvius.ink_detection.paris4_3d_gambiarra
```

Commands read the public PHercParis4 level-0 CT by default. The array is 2.4 µm, uint8, in ZYX order:
`https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0/`.
Use `--ct /path/to/masked.zarr/0` to read a local copy of the same array.

The teacher uses 2.4 µm crops. For the student, the CLI normalizes each level-0 crop, then averages each 2×2×2 block to produce 4.8 µm input.

For training, download the [DINO checkpoint](https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500) and pass its local path as `--ssl-checkpoint`. Student training also needs the teacher checkpoints produced below. Inference needs only `model.safetensors` and the CT.

## Run inference

Start with a small region:

```bash
WEIGHTS=/path/to/model.safetensors
python -m "$P.infer_volume" --weights "$WEIGHTS" --out runs/box --unit 256 \
    --box 32381 7712 8064 32509 7840 8192
```

`--box` takes `z0 y0 x0 z1 y1 x1` in 4.8 µm output voxels, with exclusive upper bounds. It selects whole work units that overlap the box, so computation can extend beyond the box. The output keeps the full-volume shape and global coordinates.

For the whole scroll on one GPU:

```bash
python -m "$P.infer_volume" --weights "$WEIGHTS" --out runs/paris4-ink \
    --unit 2048 --batch 16
```

To use two GPUs, run one shard per device in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 python -m "$P.infer_volume" --weights "$WEIGHTS" --out runs/paris4-ink \
    --unit 2048 --batch 16 --shard 0/2
CUDA_VISIBLE_DEVICES=1 python -m "$P.infer_volume" --weights "$WEIGHTS" --out runs/paris4-ink \
    --unit 2048 --batch 16 --shard 1/2
```

Use the same weights, CT, output directory, settings, box, and shard count for both processes. Each process handles its assigned units. Run one process per shard and wait for both to finish.

Rerunning verifies completed units and recomputes missing or damaged output.

Use a new output directory when changing weights or inference settings. The filesystem must support atomic renames and hard links.

`--unit` sets the work-unit edge in output voxels. It must be a multiple of `--chunk`, which defaults to 256. Full-volume inference used H100 GPUs, `--unit 2048 --batch 16`, bf16 autocast, and no test-time augmentation. Reduce `--unit` if GPU or host memory is insufficient. Batch size, unit size, hardware, and numerical kernels can affect the output codes.

### Preprocessing, blending and output

For each 128³ level-0 crop, compute p1 / p99 from its surrounding 256³ level-0 parent, clip and scale the crop to [0, 1], then average-pool by 2. The U-Net receives 64³ voxels at 4.8 µm and returns 64³ logits. Training and full-volume inference used this preprocessing.

Crops follow a global grid with nominal stride 32 output voxels, adjusted to the volume dimensions. Overlapping logits are blended with a Gaussian window (σ = 8 output voxels), using float64 accumulators. Only crops whose 256³ parent fits inside the CT are used.

The output at `<out>/code.zarr/0` is a Zarr v2 uint8 array in ZYX order. Its shape is `ceil(level0_shape / 2)`, with default chunks of 256³. Output index `j` maps to level-0 centre coordinate `2*j + 0.5` on each axis.

```text
code = clip(floor(10 * blended_logit) + 128, 1, 255)
code = 0 where there is no prediction
```

Zero marks absent CT chunks or locations outside crop coverage. Codes 2–254 represent logit intervals `[(code - 128)/10, (code - 127)/10)`; 1 and 255 include clipped tails. These codes store logits. Public display uses transformed uint8 ink scores. `sigmoid(logit)` is an ink score, not a calibrated probability. No decision threshold is supplied.

## Training stages

The stages below document how we developed the released checkpoint. They record the experiments and continuations used during discovery, not a recipe we recommend repeating. One DINO fine-tuning run followed by one student distillation run would likely be enough, but we have not tested that shorter recipe.

The teacher starts from the DINO checkpoint above, pretrained on CT that included PHercParis4. The student starts from random initialization with seed 27.

Early stages used a training and evaluation split. The final teacher and student trained on all 9,627 samples. The commands below use the complete JSON in every stage. Teacher learning rates are listed as head and backbone rates.

| Stage | Initialization | Updates | Learning rate and main changes |
|---|---|---|---|
| Teacher `linear20k` | SSL backbone + linear head | 20,000 | All backbone blocks and the head train. 0.01 / 3e-6; warmup 50, cosine over a 40,000-update schedule, stopped at 20,000. Random 3D rotations, contrast and gamma changes, blur and noise. |
| Teacher `scribble5k` | `linear20k` + new residual decoder | 5,000 | BCE on scribble samples + derived tube labels. 3e-4 / 1e-6; head warmup 50, backbone frozen 250 then ramped over 50; cosine to 0.1×. Axis permutations, flips, contrast and gamma. |
| Teacher `scribble10k` | Continues `scribble5k`, retaining optimizer and sampling state | 5,000 more, to 10,000 | Same losses; cosine from 3e-5 / 1e-7 to 0.1×, without warmup. |
| Teacher `all_aug` | Continues `scribble10k`; refreshed sampling decks | 5,000 | Re-warmup 50 to 3e-4 / 1e-6, then cosine to 0.1×. Rotation, zoom, brightness, contrast, gamma, blur and noise. |
| Student 1 | Seed-27 U-Net + `scribble10k` targets | 20,000 | 3e-4; warmup 50, cosine to 3e-5. |
| Student 2 | Restores student 1 weights, AdamW and unlabelled stream; new labelled stream + `all_aug` targets | 5,000 more, to 25,000 | Constant 3e-5. The released checkpoint is update 25,000. |

Each student update accumulates 16 labelled and 16 unlabelled views, in microbatches of 8. Its loss is class-balanced BCE on scribble samples + 4 × labelled-view distillation BCE + 4 × unlabelled-view distillation BCE. Each distillation term averages over lattice sites and then its 16 views. The student uses axis permutations, flips, contrast and gamma; tube and surface losses are off. Both models use AdamW with weight decay 0.01 and gradient-norm clipping at 1.

Teacher stages 2 to 4 also use derived tube labels. They cover voxels within one voxel of a scribble sample or a segment between same-class samples at most five voxels apart. Conflicting labels are excluded. The student does not use this loss.

### Run the six stages

Both trainers download the scribbles into the Hugging Face cache by default. To use a local copy, add `--scribbles /path/to/paris4_ink_scribbles.vc3d.json` to each command.

```bash
SSL=/path/to/dinovol_v2_ps8_with_paris4_352500.pt

python -m "$P.train_teacher" --stage linear20k --ssl-checkpoint "$SSL" \
    --out runs/linear20k
python -m "$P.train_teacher" --stage scribble5k --ssl-checkpoint "$SSL" \
    --out runs/scribble5k --init runs/linear20k/step-020000.pth
python -m "$P.train_teacher" --stage scribble10k --ssl-checkpoint "$SSL" \
    --out runs/scribble10k --init runs/scribble5k/step-005000.pth
python -m "$P.train_teacher" --stage all_aug --ssl-checkpoint "$SSL" \
    --out runs/all_aug --init runs/scribble10k/step-010000.pth

python -m "$P.train_student" --stage 1 --ssl-checkpoint "$SSL" \
    --teacher runs/scribble10k/step-010000.pth --out runs/student1
python -m "$P.train_student" --stage 2 --ssl-checkpoint "$SSL" \
    --teacher runs/all_aug/step-005000.pth --out runs/student2 \
    --init runs/student1/train-state-020000.pth
```

Use a fresh output directory for each stage. To resume, repeat the stage arguments and replace `--init`, if present, with `--resume`. Use `step-XXXXXX.pth` for a teacher or `train-state-XXXXXX.pth` for a student.

Resume requires the same CT, scribbles, SSL checkpoint, and, for students, pool and teacher checkpoint. Keep local CT data unchanged at the same path. Use historical checkpoints as `--init` for the next stage, rather than `--resume`.

Each stage writes checkpoints and `curve.jsonl`. `COMPLETE.json` marks a finished stage. Resume also repairs an interrupted student export or completion marker. The final student export is `runs/student2/step-025000.safetensors`, in the same format as `model.safetensors`.

## Scribbles

The VC3D JSON contains 9,627 labeled locations from scribbles and individual clicks, with 2,929 ink and 6,698 no-ink samples. The `ink` and `noink` collection names supply the labels. Coordinates are fractional XYZ indices in the 2.4 µm level-0 CT. Every stage uses all samples.

## License and attribution

Licensed under [MIT](https://github.com/ScrollPrize/villa/blob/ink-paris4-3d-gambiarra/LICENSE). The model uses CT data, annotations, and software from the [Vesuvius Challenge](https://scrollprize.org/) and the public [DINO volume checkpoint](https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500).
