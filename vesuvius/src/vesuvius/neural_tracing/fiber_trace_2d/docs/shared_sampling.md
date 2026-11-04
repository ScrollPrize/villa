# Sampling coordinates

Coordinate maps and augmentation maps use `vc_reader.torch.sample_array` on CPU
and CUDA. Map channels retain their XYZ/ZYX meaning; queries are supplied in
array index order (YX for images), without normalized Torch coordinates.
Sampling uses vc_reader defaults: linear float32 interpolation with zero-valued
outside neighbors, and half-up ties for nearest validity masks. Application
validity and ragged-length masks are applied after sampling.
