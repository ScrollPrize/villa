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
        # Find target keys (exclude 'image' and 'is_unlabeled')
        target_keys = [k for k in data_dict.keys() if k not in ['image', 'is_unlabeled']]
        
        # Process each target
        for target_key in target_keys:
            seg_all = data_dict[target_key].numpy()
            # Add tubed skeleton GT
            bin_seg = (seg_all > 0)
            seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)

            # Skeletonize
            if not np.sum(bin_seg[0]) == 0:
                # skel = skeletonize(bin_seg[0], surface=True)
                skel = np.zeros_like(bin_seg[0])
                Z, Y, X = skel.shape

                for z in range(Z):
                    skel[z] |= skeletonize(bin_seg[0][z])

                # for y in range(Y):
                #     skel[:, y, :] |= skeletonize(bin_seg[0][:, y, :])
                #
                # for x in range(X):
                #     skel[:, :, x] |= skeletonize(bin_seg[0][:, :, x])

                skel = (skel > 0).astype(np.int16)
                if self.do_tube:
                    skel = dilation(dilation(skel))
                if self.do_open:
                    skel = opening(skel)
                if self.do_close:
                    skel = closing(skel)
                skel *= seg_all[0].astype(np.int16)
                seg_all_skel[0] = skel

            # Store skeleton for each target with a unique key
            data_dict[f"{target_key}_skel"] = torch.from_numpy(seg_all_skel)
        
        return data_dict