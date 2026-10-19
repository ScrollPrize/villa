this began as a straight copy of batchgeneratorsv2, with some additional stuff added and with more flexibility to fit our repo.
https://github.com/MIC-DKFZ/batchgeneratorsv2

An opt-in C++ backend is available through `vc_reader.transforms`. `ZarrDataset`
accepts `augmentation=pipeline, augmentation_seed=42`; `native.py` provides
`AugmentationSampler` for reproducible epochs with persistent workers and
`AugmentedDataset` for other dictionary-returning datasets. The existing backend
remains the default. See the [transform guide](../../../../../vc_reader/README.md)
for the API and target declarations.

