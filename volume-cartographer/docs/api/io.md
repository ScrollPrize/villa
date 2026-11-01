# I/O API

This document covers file I/O utilities for image and volume data.

## Image files

TIFF, PNG and JPEG are read and TIFF is written by vc_reader (`vc_reader/image_write.hpp`,
`vcr::sources`). `vc/core/util/ImageMat.hpp` adapts them to `cv::Mat`: `readImageMat`,
`writeTiffMat`.

## Zarr volumes

volume-cartographer has no Zarr implementation of its own: open, read, sample and
write volumes through vc_reader (`vc_reader/zarr.hpp`, `vc_reader/array.hpp`,
`vc_reader/grid.hpp`). `vc/core/types/Volume.hpp` says where a volume's source is;
see [Volume](Volume.md).
