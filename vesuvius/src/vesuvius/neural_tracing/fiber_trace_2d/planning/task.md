# Contrastive Embedding With Cosine Similarity

Implement the todo item "contrastive embedding using cosine similarity":

- change CP sampling for training so each contrastive batch uses `N` CPs from the same fiber repeated `M` times to fill the batch;
- add a simple embedding head and cosine-similarity contrastive loss;
- positive supervision compares CP-neighborhood pixels from the same fiber;
- negative supervision compares CP-neighborhood pixels to other valid pixels from the batch with balanced positive/negative weighting;
- keep direction supervision active;
- visualize per-pixel embedding similarity against the patch CP embedding in TensorBoard;
- keep geometric augmentations independent across repeated fiber patches and synchronize value/image augmentations for contrastive groups.
