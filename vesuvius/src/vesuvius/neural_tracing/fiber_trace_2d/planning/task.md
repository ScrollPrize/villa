# Contrastive Similarity-Mean Sparsity Loss

Add another contrastive embedding loss term: for each supervised CP embedding,
the average normalized embedding-similarity image against that CP embedding
should be `0.1`.

The normalized similarity image is the same `0..1` space used by TensorBoard
embedding-similarity visualization: `0.5 + 0.5 * cosine_similarity`.
This should push CP-similar embeddings toward a small sparse region instead of
allowing broad high-similarity areas.
