# `surface-labels`

Labeled papyrus surfaces: voxelized, recto (inside) sheet surfaces within the
scroll volume.

Surface segmentation separates the volumetric part of the papyrus we want to
map from the rest of the scan. This dataset provides labels of the recto
surfaces of the papyrus sheet, paired with the corresponding scroll volume
data, so they can be used directly as training inputs for machine-learning
models that segment the surface.

## Contents

| Path | Description |
| --- | --- |
| _(volume)_ | Scroll volume data for each labeled region. |
| _(labels)_ | Voxelized recto-surface labels, aligned to the volume. |

<!-- TODO: replace the rows above with the real folder/file layout once the
     dataset is uploaded (e.g. per-scroll or per-cube subfolders, file format
     such as .nrrd / TIFF stack / nnU-Net imagesTr+labelsTr). -->

## Conventions

- Labels are volumetric (voxelized) and registered to the paired scroll volume.

## License

See <https://dl.ash2txt.org/LICENSE.txt>.
