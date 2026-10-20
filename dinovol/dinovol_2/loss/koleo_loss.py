import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather


logger = logging.getLogger("dinov2")


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self, *, distributed: bool = True, loss_group_size: int | None = None):
        super().__init__()
        self.distributed = distributed
        if loss_group_size is not None and (isinstance(loss_group_size, bool) or not isinstance(loss_group_size, int) or loss_group_size < 2):
            raise ValueError("KoLeo loss_group_size must be an integer >= 2 or None")
        self.loss_group_size = loss_group_size
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x, candidates=None, offset=0, group_size=None):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        if candidates is None:
            candidates = x
        dots = torch.mm(x, candidates.t())
        n = x.shape[0]
        rows = torch.arange(n, device=x.device)
        if group_size is not None:
            # Contiguous groups in rank-major order, independently for each view.
            groups = torch.div(rows + offset, group_size, rounding_mode="floor")
            candidate_groups = torch.div(torch.arange(candidates.shape[0], device=x.device), group_size, rounding_mode="floor")
            dots.masked_fill_(groups[:, None] != candidate_groups[None, :], -torch.inf)
        dots[rows, rows + offset] = -torch.inf
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.amp.autocast(device_type=student_output.device.type, enabled=False):
            student_output = F.normalize(student_output.float(), eps=eps, p=2, dim=-1)
            candidates = student_output
            offset = 0
            if self.distributed and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                # Training loaders use equal per-rank batches with drop_last=True.
                # Autograd gathering lets remote nearest neighbours receive gradients.
                candidates = torch.cat(all_gather(student_output), dim=0)
                offset = dist.get_rank() * student_output.shape[0]
            if candidates.shape[0] < 2:
                return student_output.sum() * 0.0
            group_size = self.loss_group_size
            if group_size is not None:
                group_size = min(group_size, candidates.shape[0])
                if candidates.shape[0] % group_size:
                    raise ValueError(
                        f"KoLeo comparison batch ({candidates.shape[0]}) must be divisible by "
                        f"loss_group_size ({group_size}); choose a divisor or set koleo_loss_group_size=null."
                    )
            with torch.no_grad():
                I = self.pairwise_NNs_inner(student_output, candidates, offset, group_size)  # noqa: E741
            distances = self.pdist(student_output, candidates[I])
            loss = -torch.log(distances + eps).mean()
        return loss
