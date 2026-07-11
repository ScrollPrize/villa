# Switch Direction Model Normalization To BatchNorm

Todo item:

- GroupNorm with a single supervision sample per patch can let the model use
  patch-global statistics in a way that behaves like a non-convolutional
  classifier. Switch the default 2D direction model from GroupNorm to a
  BatchNorm-style normalization.
