# Parting

Parting tells the same papyrus layer from the next one where two layers touch. At a contact it takes a point `a` on one traced layer, a second point `a2` on the same layer and a point `b` on the next layer at the same distance, and scores the pair (a, a2) and the pair (a, b); a higher score means "same layer". Tracers switch sheets at exactly such places, where two layers press into one thick band.

It is a small classifier (a soft vote of gradient boosting and a one-layer MLP) on 669 numbers per point pair: the released base dinovol checkpoint's tokens at the two points (`scrollprize/dinovol_ps8_9.6um_101626`, `teacher_305k.pt`), their token similarity along the line between them, CT profiles along each point's normal and along that line, and base dinovol's own token cosine. That cosine alone is the control every result is compared against. Parting scores given point pairs; it does not trace and does not read ink. Its scores are model output.

## Results so far

Each test was fixed before any of the object's 3D CT was read for it, on fragments that neither Parting's model nor base dinovol trained on, against the Vesuvius Challenge team's traced layer segments. The bar: touching AUC at least 0.75 and a gain over the base dinovol cosine of at least +0.05 whose 2.5 % bootstrap bound (20,000 resamples of layer pairs) is above 0.

| Fragment (scroll) | Touching sites, layer pairs | Parting AUC | Base dinovol cosine AUC | Gain (2.5 % bound) | Verdict |
| --- | --- | --- | --- | --- | --- |
| PHerc1667Cr1Fr3 (PHerc1667) | 203, 7 | 0.960 | 0.915 | +0.045 (+0.020) | FAIL |
| PHercParis1Fr34, Frag3 (PHerc. Paris 1) | 150, 5 | 0.895 | 0.839 | +0.056 (+0.037) | PASS |
| PHercParis2Fr143 (PHerc. Paris 2) | 270, 9 | 0.975 | 0.965 | +0.010 (-0.0005) | FAIL |

One pass in three. Where base dinovol is already near 0.97 there is little room for a gain. At fused contacts (under 3 voxels apart) Parting adds little (0.633 against 0.611 on 38 Frag3 sites). The fragments are flat layer stacks, not rolled scrolls. The model behind these numbers was trained on contact sites from the published hand-traced windings of PHerc0139 and PHerc0172. Full record: [Scrolls test log](https://github.com/ChaseHendrick/Scrolls/blob/main/docs/logs/2026-10-29-parting-fragment-contact.md).

## Model file

The trained model behind every result above is [`SharpMeow/parting-v1`](https://huggingface.co/SharpMeow/parting-v1) on Hugging Face (`parting-v1.joblib`, 4.1 MB, sha256 `f0684ab2ec5725ad9050e08bcdf73df22983960ba906d222cb9c8068a0fdc601`, CC BY-NC 4.0 like the scans it was trained on). A joblib file runs pickled code when loaded, so `score` loads only that file unless you pass `--trust-model` for one you made with `fit`. Scans and traced layers are read live from the Vesuvius Challenge open data (CC BY-NC 4.0). The base dinovol checkpoint is downloaded from Hugging Face at run time, pinned by revision and sha256, and loaded with `torch.load(weights_only=True)`.

## Run it headless

From `dinovol/`, after `uv sync --extra parting` (adds scikit-learn, joblib and huggingface-hub):

```bash
# 1. contact sites from traced layers, in stack order (geometry only, no CT)
python -m dinovol_2.eval.parting sites --layer L0/pointset.vcps --layer L1/pointset.vcps --shape Z Y X --out sites.json
# 2. features per site: one 128^3 CT window each, read from a public zarr v2 level with 128^3 uint8 chunks
python -m dinovol_2.eval.parting features --sites sites.json --volume ZARR_LEVEL_URL --device cuda --out run
#    a finer scan can be box-averaged to about 8 um, as the tests did: --resample 3.24:7.91
# 3. score with the released model and the gate
MODEL=$(python -c "from huggingface_hub import hf_hub_download as d; print(d('SharpMeow/parting-v1', 'parting-v1.joblib'))")
python -m dinovol_2.eval.parting score --run run --model "$MODEL" --sites sites.json --verdict --out readout.json
#    or fit your own on runs from other layers (touching sites only) and score with --trust-model
python -m dinovol_2.eval.parting fit --run train_run --out mine.joblib
# largest per-site differences between two feature runs (devices, code versions)
python -m dinovol_2.eval.parting compare run_a run_b
# the published Frag3 site list, rebuilt from public data and checked against its hash (CPU, about 1 GB download)
python -m dinovol_2.eval.parting frag3 --out frag3 --sites-only
# the Frag3 test end to end with the released model, reported against the frozen numbers
python -m dinovol_2.eval.parting frag3 --out frag3 --model "$MODEL" --device cuda
```

Layers are `.vcps` ordered pointsets, `.npy` (H, W, 3) xyz grids or tifxyz folders, in xyz voxels of the volume read; give them in stack order and use `--scale`, `--row-step` and `--col-step` to match the volume's voxel size and thin the grid. `sites --exclude-crossings VOX` keeps windows away from places where consecutive traces cross. Every command writes JSON; `score` prints `primary`, `permutation_gate` and `verdict` (PASS, FAIL or VOID; `--verdict` applies the gate). A run folder holds `results.jsonl`, `features.npz` (float16) and `run.json`, and `features` resumes a folder it already started.

Measured: features for 283 Frag3 sites took about 4 minutes on one RTX PRO 6000 (bf16); on CPU one window takes about 5 s on 4 cores (float32). The Frag3 site rebuild took 4 minutes on a 4-core CPU.

## Encoder

`encoder.py` builds the backbone from the checkpoint's own model config with `DinoVitStudentTeacher._build_backbone`, loads the teacher backbone weights strictly, and normalizes each window with `dinovol_2.dataset.normalization.normalize_robust` (clip to the 1st and 99th percentiles, subtract the median, divide by 1.4826 times the MAD), the dataset default the checkpoint was pretrained with, since its config names no `normalization_scheme`. On CUDA it runs under bf16 autocast, as the published runs did; on CPU in float32.

The published runs built the same backbone from a vendored copy of this dinovol code ([jimmylomro/hercUNet](https://github.com/jimmylomro/hercUNet) at `de4acb7`). On CPU (torch 2.14.1) this port gives bit-identical tokens to that copy on three real Frag3 windows (two box-averaged to 7.91 um, one native at 6.48 um): max abs difference 0.0, normalized input max abs difference 0.0, and identical features and scores on two sites through `features` and `compare`.

## Tests

```bash
python -m pytest tests/test_parting.py tests/test_parting_encoder.py
```

They use synthetic data and a tiny random checkpoint: no network, no GPU, no released weights. The scikit-learn test is skipped when the `parting` extra is not installed.

## License

Contributed by Chase Hendrick under the dinovol [MIT License](../../../LICENSE); the same code is also published under Apache-2.0 in [ChaseHendrick/Scrolls](https://github.com/ChaseHendrick/Scrolls). It contains no third-party code. The base dinovol checkpoint's model page states no license, so it is downloaded from its own host and never redistributed.
