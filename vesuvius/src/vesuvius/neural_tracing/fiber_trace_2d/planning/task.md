# Cross-Fiber Contrastive CP Negatives

Add an additional contrastive negative term for embedding training: each CP
embedding in a batch should be penalized for high cosine similarity to CP
embeddings from other fibers in the same batch.

This term should use the same cosine margin as the existing non-CP pixel
negative term. When both negative terms are available, they should split the
negative half of the balanced contrastive objective equally, so the overall
positive-vs-negative balance stays unchanged.
