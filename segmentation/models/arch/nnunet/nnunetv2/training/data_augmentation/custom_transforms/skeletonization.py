from typing import Tuple

import torch
import numpy as np
from skimage.morphology import skeletonize, dilation

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class SkeletonTransform(BasicTransform):
    def __init__(self, do_tube: bool = True, ignore_label: int = None):
        """
        Calculates the skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.do_tube = do_tube
        self.ignore_label = ignore_label
    
    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].numpy()
        if self.ignore_label is not None:
            seg_all = np.where(seg_all == self.ignore_label, 0, seg_all)
        # Add tubed skeleton GT
        bin_seg = (seg_all > 0)
        seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
        
        # Skeletonize
        if not np.sum(bin_seg[0]) == 0:
            skel = skeletonize(bin_seg[0])
            skel = (skel > 0).astype(np.int16)
            if self.do_tube:
                skel = dilation(dilation(skel))
            skel *= seg_all[0].astype(np.int16)
            seg_all_skel[0] = skel

        data_dict["skel"] = torch.from_numpy(seg_all_skel)
        return data_dict
        
class MedialSurfaceTransform(BasicTransform):
    def __init__(self, do_tube: bool = True, ignore_label: int = None):
        """
        Calculates the medial surface skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.do_tube = do_tube
        self.ignore_label = ignore_label
    
    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].numpy()
        if self.ignore_label is not None:
            seg_all = np.where(seg_all == self.ignore_label, 0, seg_all)
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
            #     skel[:, y, :] |= skeletonize(bin_seg[0][:, y, :], surface=False)
            #
            # for x in range(X):
            #     skel[:, :, x] |= skeletonize(bin_seg[0][:, :, x], surface=False)
            
            skel = (skel > 0).astype(np.int16)
            if self.do_tube:
                skel = dilation(dilation(skel))
            skel *= seg_all[0].astype(np.int16)
            seg_all_skel[0] = skel

        data_dict["skel"] = torch.from_numpy(seg_all_skel)
        return data_dict
