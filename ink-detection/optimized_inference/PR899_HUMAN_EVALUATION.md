# PR #899 Human Evaluation Notes

This note records the manual review behind the Primus optimized-inference
loader. It exists to make the contribution auditable before responding to the
reviewer feedback on PR #899.

## Why Primus Belongs Here

Villa already supports LeJEPA pretraining and Primus-backed fine-tuning in the
training stack. The optimized inference container is the canonical submission
path, so it should be able to load the same fine-tuned model family without
requiring a custom downstream inference script.

The useful contract is narrow:

- `MODEL_TYPE=primus` resolves through the same runtime contract gate as
  TimeSformer and ResNet3D.
- `entrypoint.py` dispatches to a model-specific `load_model` function.
- The model wrapper exposes `forward`, `get_output_scale_factor`, `eval`, and
  `to`, matching the existing inference-model protocol.
- The wrapper receives optimized-inference tiles as `(B, 1, D, H, W)` and keeps
  that shape for the Villa `NetworkFromConfig` model.

## Manual Findings

- The original loader had a misleading double permutation in `forward`; it
  returned the tensor to its original shape, but made the shape contract harder
  to audit. The wrapper now preserves the optimized-inference volume shape
  directly.
- The previous PR narrative correctly identified the missing `vesuvius`
  dependency, but leaving that as a caveat made the container support claim too
  weak. The next repair slice should either install the local `villa/vesuvius`
  package in the GPU image or explicitly narrow the PR until a container smoke
  test exists.
- The new unit tests exercise the wrapper shape contract and checkpoint loading
  against a stubbed `NetworkFromConfig`, so they verify the PR's local logic
  without requiring a heavyweight Primus checkpoint in CI.

## Evidence Added

Run from `ink-detection/optimized_inference`:

```bash
python -m unittest tests.test_runtime_contracts tests.test_model_primus -v
```

This validates:

- `MODEL_TYPE=primus` remains accepted by runtime contracts.
- 4D tiles are lifted to `(B, 1, D, H, W)`.
- 5D optimized-inference volumes keep `(B, 1, D, H, W)`.
- checkpoint config normalization provides `train_patch_size`.
- the ConfigManager shim exposes the attributes used by Villa's real
  `NetworkFromConfig`.
- the loader selects the `ink` target when present.

After installing the local Villa package into the Vesuvius Autoresearch virtual
environment:

```bash
uv pip install --python /home/jon/openclaw-workspace/Neo-VM/projects/vesuvius-autoresearch/.venv/bin/python -e '/home/jon/openclaw-workspace/Neo-VM/projects/vesuvius-autoresearch/villa/vesuvius[models]'
python -m unittest tests.test_model_primus_integration -v
```

The integration test builds a minimal real `NetworkFromConfig` Primus-S model,
saves a production checkpoint envelope, reloads it through `model_primus`, and
checks that a `(1, 1, 16, 16, 16)` optimized-inference tensor produces an
`ink` output of the same shape.

## Still Required Before Reviewer Reply

- Resolve the Docker dependency path for `vesuvius`.
- Run an end-to-end optimized inference smoke test with a real or minimal
  Primus checkpoint envelope inside the container.
- Rewrite the PR description/comment in human terms and remove generated-content
  footers.
