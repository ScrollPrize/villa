# Regional flat ink output

`--roi ROW_START ROW_STOP COL_START COL_STOP` writes only the requested rectangle and a `.roi.json` parent-coordinate sidecar. Bounds are half-open rows/columns at the selected resolution. The original global CT tiles, normalization and blend schedule remain; only output contributions are clipped.

```bash
python -m vesuvius.ink_detection.inference.infer surface.zarr step-075000.pth roi.tif \
  --roi 2328 2408 2712 2792 --stride 64 --blend-mode hann \
  --batch-size 1 --num-workers 0 --no-compile
```

This produces an 80×80 TIFF. The sidecar records source resolution, parent dimensions, ROI bounds, selected layers and tiling. Coordinates are not XYZ or physical micrometers. Reusing the output path for a full-frame image invalidates old ROI metadata before replacing the TIFF.

The existing full-parent mask workflow already gives identical regional pixels and the same model tile count. The addition avoids preparing a full-frame mask and writing/cropping a full-frame output. In two real CLI examples, TIFF plus sidecar is 5,133 versus 415,490 bytes and 5,258 versus 270,633 bytes. These are file-size measurements, not peak-RAM or speed measurements.

Full [real-data comparisons, images, protocols and limitations](https://github.com/RhudeJJJJ/villa/blob/ink-output-roi-evidence-20261020/vesuvius/docs/ink-output-roi/README.md) are on the separate evidence branch. Eight native/public-reference regions show the bounded context benefit. The separate independent-IR test FAILED; no independent physical ink improvement or new readable letters is claimed. PHerc.139 is model-seen; PHerc.841 labels have unknown original annotation-scan provenance. Existing normalization work #1481/#1483/#1646/#1773 and global-grid ROI work #1247 are credited.

The real CT/saved-logit fixture in `tests/ink_detection/fixtures` checks assembly and output lifecycle. Its CT data retain Vesuvius Challenge source attribution and CC-BY-NC 4.0 terms unless asset-specific terms say otherwise; the software remains under the repository MIT license. No restricted legacy Fragment1 images are included.

Run the existing flat-inference/boundary tests and `test_roi_inference.py`, `test_roi_output_metadata.py`. Recorded v2 validation: 59 tests and two targeted actual-checkpoint CLI processes pass on Python 3.12/PyTorch 2.9.1 CPU/Windows. This does not certify all GPUs, TTA or distributed batches. The actual checkpoint and reproducible source/frame metadata are pinned in the evidence guide; no model weights are changed or included.
