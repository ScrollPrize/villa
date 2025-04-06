from typing import Tuple

import torch
import numpy as np
from skimage.morphology import skeletonize, dilation

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class SkeletonTransform(BasicTransform):
    def __init__(self, do_tube: bool = True):
        """
        Calculates the skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.do_tube = do_tube
    
    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].numpy()
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
    def __init__(self, do_tube: bool = True):
        """
        Calculates the medial surface skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.do_tube = do_tube
    
    def apply(self, data_dict, **params):
        seg_all = data_dict['segmentation'].numpy()
        # Add tubed skeleton GT
        bin_seg = (seg_all > 0)
        seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
        
        # Skeletonize
        if not np.sum(bin_seg[0]) == 0:
            skel = skeletonize(bin_seg[0], surface=True)

            # initializing different axes
            skel_x = np.zeros_like(skel)
            skel_y = np.zeros_like(skel)
            skel_z = np.zeros_like(skel)

            for z in range(skel.shape[0]):
                skel_z[z] = skeletonize(bin_seg[0][z], surface=False)
            for y in range(skel.shape[1]):
                skel_y[:,y,:] = skeletonize(bin_seg[0][:,y,:], surface=False)
            for x in range(skel.shape[2]):
                skel_x[:,:,x] = skeletonize(bin_seg[0][:,:,x], surface=False)

            np.logical_or(skel, skel_z, out=skel)
            np.logical_or(skel, skel_y, out=skel)
            np.logical_or(skel, skel_x, out=skel)
            
            skel = (skel > 0).astype(np.int16)
            if self.do_tube:
                skel = dilation(dilation(skel))
            skel *= seg_all[0].astype(np.int16)
            seg_all_skel[0] = skel

        data_dict["skel"] = torch.from_numpy(seg_all_skel)
        return data_dict
