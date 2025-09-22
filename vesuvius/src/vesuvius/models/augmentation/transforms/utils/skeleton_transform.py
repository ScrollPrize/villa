import torch
from skimage.morphology import skeletonize, dilation, opening, closing
import numpy as np

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform


class MedialSurfaceTransform(BasicTransform):
    def __init__(self,
                 do_tube: bool = True,
                 do_open: bool = False,
                 do_close: bool = True,):
        """
        Calculates the medial surface skeleton of the segmentation (plus an optional 2 px tube around it)
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.do_tube = do_tube
        self.do_open = do_open
        self.do_close = do_close

    def apply(self, data_dict, **params):
        # Collect regression keys to avoid processing continuous aux targets
        regression_keys = set(data_dict.get('regression_keys', []) or [])
        # Find eligible target keys: tensor-valued, not image/meta, not regression aux
        target_keys = [
            k for k, v in data_dict.items()
            if k not in ['image', 'is_unlabeled', 'regression_keys']
            and isinstance(v, torch.Tensor)
            and k not in regression_keys
        ]

        # Process each target
        for target_key in target_keys:
            t = data_dict[target_key]
            orig_device = t.device
            seg_all = t.detach().cpu().numpy()

            bin_seg = seg_all > 0
            seg_all_skel = np.zeros_like(bin_seg, dtype=np.float32)

            for c in range(bin_seg.shape[0]):
                seg_c = bin_seg[c]
                if seg_c.sum() == 0:
                    continue

                if seg_c.ndim == 3:
                    skel = np.zeros_like(seg_c, dtype=bool)
                    for z in range(seg_c.shape[0]):
                        skel[z] |= skeletonize(seg_c[z])
                elif seg_c.ndim == 2:
                    skel = skeletonize(seg_c)
                else:
                    raise ValueError(f"Unsupported segmentation dimensionality {seg_c.ndim} for skeletonization")

                # for y in range(Y):
                #     skel[:, y, :] |= skeletonize(bin_seg[0][:, y, :])
                #
                # for x in range(X):
                #     skel[:, :, x] |= skeletonize(bin_seg[0][:, :, x])

                skel = (skel > 0).astype(np.float32)
                if self.do_tube:
                    skel = dilation(dilation(skel))
                if self.do_open:
                    skel = opening(skel)
                if self.do_close:
                    skel = closing(skel)
                skel = skel.astype(np.float32) * seg_all[0].astype(np.float32)
                seg_all_skel[0] = skel

                seg_all_skel[c] = (skel.astype(np.float32) * seg_all[c].astype(np.float32))

            data_dict[f"{target_key}_skel"] = torch.from_numpy(seg_all_skel).to(orig_device)

        return data_dict
