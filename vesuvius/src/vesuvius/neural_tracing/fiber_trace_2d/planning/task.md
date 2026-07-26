# 3D Direction-Conditioned Recurrent Decoder Plan

Plan a replacement for the current two-free-branch 3D fiber direction model
with a direction-conditioned recurrent decoder.

The desired model shape is:

- A shared 3D U-Net encodes the CP-centered volume patch into a per-voxel latent
  feature volume.
- The U-Net latent width passed to the conditioned decoder is configured
  separately from the U-Net starting width and defaults to `64` channels.
- The conditioned decoder receives `latent_width + 6` channels at each voxel:
  the U-Net latent channels plus a six-channel Lasagna 3x2 query direction
  encoding.
- The conditioned decoder is pointwise only: no spatial 3D layers. It may use a
  small stack of `1x1x1`/per-voxel layers.
- The conditioned decoder outputs one seven-channel prediction per query:
  six Lasagna 3x2 direction channels plus one presence channel.

Training semantics:

- The all-zero six-channel query is reserved as an off-manifold unconditioned
  query token. It must not be decoded as a real direction.
- Positive supervised locations use two positive queries with equal weight:
  - the zero/unconditioned query predicts positive presence and the GT
    direction;
  - one sampled direction approximately perpendicular to the GT direction,
    with configurable jitter defaulting to a `45` degree range, predicts
    positive presence and the same GT direction.
- Negative presence queries are produced over all valid locations using both
  the zero query and random direction queries. They use the existing margin
  handling and weighted BCE semantics; weak negative terms at positive pixels
  are intentional and mathematically act as a softened positive target under
  the chosen weight ratio.
- Direction loss remains weighted MSE on the six encoded Lasagna direction
  channels at positive query samples.
- Presence loss remains BCE on sigmoid probabilities unless the implementation
  deliberately changes the model to return logits and uses
  `BCEWithLogitsLoss`.

Inference semantics:

- First decode with the zero query to get the strongest local fiber direction.
- Subsequent recurrent decode steps feed the previous decoded direction encoded
  with `encode_lasagna_direction_3x2(...)` as the query.
- The intended learned behavior is that a non-empty direction query asks for the
  strongest local direction not already explained by that query.
