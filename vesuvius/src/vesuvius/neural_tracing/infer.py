
import json
import zarr
import cc3d
import torch
import accelerate
import numpy as np

from vesuvius.neural_tracing.dataset import get_crop_from_volume, build_localiser, make_heatmaps
from vesuvius.models.run.tta import infer_with_tta


class Inference:

    def __init__(self, model, config, volume_zarr, volume_scale):

        self.accelerator = accelerate.Accelerator(
            mixed_precision=config['mixed_precision'],
        )


        self.model = self.accelerator.prepare(model)
        self.device = self.accelerator.device
        self.config = config
        self.do_tta = config.get('do_tta', True)
        self.tta_type = config.get('tta_type', "rotation")
        self.tta_batched = bool(config.get('tta_batched', True))

        if config.get('compile', True):
            self.model.compile()

        self.model.eval()


        print(f"loading volume zarr {volume_zarr}...")
        ome_zarr = zarr.open_group(volume_zarr, mode='r')
        self.volume = ome_zarr[str(volume_scale)]
        with open(f'{volume_zarr}/meta.json', 'rt') as meta_fp:
            self.voxel_size_um = json.load(meta_fp)['voxelsize']
        print(f"volume shape: {self.volume.shape}, dtype: {self.volume.dtype}, voxel-size: {self.voxel_size_um * 2 ** volume_scale}um")

    def get_heatmaps_at(self, zyx, prev_u, prev_v, prev_diag):
        crop_size = self.config['crop_size']
        volume_crop, min_corner_zyx = get_crop_from_volume(self.volume, zyx, crop_size)
        use_localiser = bool(self.config.get('use_localiser', True))
        if use_localiser:
            localiser = build_localiser(zyx, min_corner_zyx, crop_size)
        prev_u_heatmap = make_heatmaps([prev_u[None]], min_corner_zyx, crop_size) if prev_u is not None else torch.zeros([1, crop_size, crop_size, crop_size])
        prev_v_heatmap = make_heatmaps([prev_v[None]], min_corner_zyx, crop_size) if prev_v is not None else torch.zeros([1, crop_size, crop_size, crop_size])
        prev_diag_heatmap = make_heatmaps([prev_diag[None]], min_corner_zyx, crop_size) if prev_diag is not None else torch.zeros([1, crop_size, crop_size, crop_size])
        input_parts = [
            volume_crop[None, None].to(self.device),
            prev_u_heatmap[None].to(self.device),
            prev_v_heatmap[None].to(self.device),
            prev_diag_heatmap[None].to(self.device),
        ]
        if use_localiser:
            input_parts.insert(1, localiser[None, None].to(self.device))
        inputs = torch.cat(input_parts, dim=1)

        def forward(model_inputs):
            outputs = self.model(model_inputs)
            logits = outputs['uv_heatmaps'] if isinstance(outputs, dict) else outputs
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return logits

        def run_with_tta(model_inputs):
            if self.do_tta:
                return infer_with_tta(forward, model_inputs, self.tta_type, batched=self.tta_batched)
            return forward(model_inputs)

        with torch.no_grad():
            logits = run_with_tta(inputs)
            logits = logits.squeeze(0).reshape(2, self.config['step_count'], crop_size, crop_size, crop_size)  # u/v, step, z, y, x
            probs = torch.sigmoid(logits)

        return probs, min_corner_zyx

    def get_blob_coordinates(self, heatmap, min_corner_zyx, threshold=0.5, min_size=8):
        # Find up to four blobs of sufficient size; return their centroids in descending order of blob size
        # TODO: strip blobs that are further than K * step_size in euclidean space
        cc_labels = cc3d.connected_components((heatmap > threshold).cpu().numpy(), connectivity=18, binary_image=True)
        cc_labels, num_ccs = cc3d.dust(cc_labels, threshold=min_size, precomputed_ccl=True, return_N=True)
        cc_stats = cc3d.statistics(cc_labels)
        centroid_zyxs = cc_stats['centroids'][1:] + min_corner_zyx.numpy()
        size_order = np.argsort(-cc_stats['voxel_counts'][1:])[:4]
        centroid_zyxs = centroid_zyxs[size_order]
        return torch.from_numpy(centroid_zyxs.astype(np.float32))
