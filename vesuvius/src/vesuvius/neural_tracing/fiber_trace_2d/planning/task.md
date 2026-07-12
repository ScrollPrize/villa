# Reachable-Area Contrastive Similarity-Mean Sparsity Loss

Restrict the contrastive similarity-mean sparsity loss to the area where CPs
can actually appear under configured shift augmentation.

This should use the same reachable rectangle used by the negative pixel
contrastive loss, so unreachable patch edges are not included in the average
similarity target.
