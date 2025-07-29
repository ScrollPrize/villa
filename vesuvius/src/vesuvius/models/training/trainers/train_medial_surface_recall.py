import torch
from skimage.morphology import skeletonize, dilation
import numpy as np

from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform
from vesuvius.models.training.loss.skeleton_recall import DC_SkelREC_and_CE_loss
from vesuvius.models.training.loss.nnunet_losses import MemoryEfficientSoftDiceLoss

class MedialSurfaceTransform(BasicTransform):
    def __init__(self, do_tube: bool = False):
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
            skel = np.zeros_like(bin_seg[0])
            Z, Y, X = skel.shape

            for z in range(Z):
                skel[z] |= skeletonize(bin_seg[0][z])

            for y in range(Y):
                skel[:, y, :] |= skeletonize(bin_seg[0][:, y, :])

            for x in range(X):
                skel[:, :, x] |= skeletonize(bin_seg[0][:, :, x])

            skel = (skel > 0).astype(np.int16)
            if self.do_tube:
                skel = dilation(dilation(skel))
            skel *= seg_all[0].astype(np.int16)
            seg_all_skel[0] = skel

        data_dict["skel"] = torch.from_numpy(seg_all_skel)
        return data_dict

class MedialSurfaceRecallTrainer(BaseTrainer):
    def __init__(self, mgr=None, verbose: bool = True):
        super().__init__(mgr, verbose)
        self.skel_transform = MedialSurfaceTransform(do_tube=False)

    def _get_model_outputs(self,
                           model,
                           data_dict,
                           autocast_ctx):

        inputs = data_dict["image"].to(self.device, dtype=torch.float32)
        targets_dict = {
            k: v.to(self.device, dtype=torch.float32)
            for k, v in data_dict.items()
            if k not in ["image", "patch_info", "is_unlabeled"]
        }

        # Apply skeleton transform to get skel from segmentation
        # The transform expects 'segmentation' key, so we need to create a temp dict
        temp_dict = {"segmentation": targets_dict.get("segmentation", targets_dict.get("ink", None))}
        if temp_dict["segmentation"] is not None:
            temp_dict = self.skel_transform.apply(temp_dict)
            targets_dict["skel"] = temp_dict["skel"].to(self.device, dtype=torch.float32)

        with autocast_ctx:
            outputs = model(inputs)

        return inputs, targets_dict, outputs

    def _compute_train_loss(self, outputs, targets_dict, loss_fns):
        """Compute training loss for all tasks."""
        total_loss = 0.0
        task_losses = {}

        for t_name, t_gt in targets_dict.items():
            if t_name == "skel":
                continue  # Skip skel as it's not a predicted output
                
            t_pred = outputs[t_name]
            task_loss_fns = loss_fns[t_name]  # List of (loss_fn, weight) tuples
            task_weight = self.mgr.targets[t_name].get("weight", 1.0)

            task_total_loss = 0.0
            for loss_fn, loss_weight in task_loss_fns:
                # Check if this is the MedialSurfaceRecall loss
                if hasattr(loss_fn, '__class__') and loss_fn.__class__.__name__ == 'DC_SkelREC_and_CE_loss':
                    # Pass skeleton data directly to the loss
                    loss_value = loss_fn(t_pred, t_gt, targets_dict.get("skel", t_gt))
                else:
                    # For non-skeleton losses, check if auxiliary loss computation would call the skeleton loss
                    # If so, bypass auxiliary loss computation
                    loss_name = loss_fn.__class__.__name__ if hasattr(loss_fn, '__class__') else str(loss_fn)
                    if 'DC_SkelREC_and_CE_loss' in loss_name:
                        # This shouldn't happen but just in case
                        loss_value = loss_fn(t_pred, t_gt, targets_dict.get("skel", t_gt))
                    else:
                        # Use default auxiliary loss computation
                        from vesuvius.models.training.auxiliary_tasks import compute_auxiliary_loss
                        loss_value = compute_auxiliary_loss(loss_fn, t_pred, t_gt, outputs,
                                                            self.mgr.targets[t_name])
                task_total_loss += loss_weight * loss_value

            weighted_loss = task_weight * task_total_loss
            total_loss += weighted_loss

            # Store the actual loss value (after task weighting but before grad accumulation scaling)
            task_losses[t_name] = task_total_loss.detach().cpu().item()

        return total_loss, task_losses

    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):
        """Compute validation loss for all tasks."""
        task_losses = {}

        for t_name, t_gt in targets_dict.items():
            if t_name == "skel":
                continue  # Skip skel as it's not a predicted output
                
            t_pred = outputs[t_name]
            task_loss_fns = loss_fns[t_name]  # List of (loss_fn, weight) tuples

            task_total_loss = 0.0
            for loss_fn, loss_weight in task_loss_fns:
                # Check if this is the MedialSurfaceRecall loss
                if hasattr(loss_fn, '__class__') and loss_fn.__class__.__name__ == 'DC_SkelREC_and_CE_loss':
                    # Pass skeleton data directly to the loss
                    loss_value = loss_fn(t_pred, t_gt, targets_dict.get("skel", t_gt))
                else:
                    # Compute loss normally
                    loss_value = loss_fn(t_pred, t_gt)
                task_total_loss += loss_weight * loss_value

            task_losses[t_name] = task_total_loss.detach().cpu().item()

        return task_losses

