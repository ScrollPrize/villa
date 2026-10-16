this began as a straight copy of batchgeneratorsv2, with some additional stuff added and with more flexibility to fit our repo.
https://github.com/MIC-DKFZ/batchgeneratorsv2

An opt-in C++ backend is available through `vc_reader.augment`. `ZarrDataset`
accepts `augmentation=pipeline, augmentation_seed=42`; `native.py` provides
`AugmentationSampler` for reproducible epochs with persistent workers and
`AugmentedDataset` for other dictionary-returning datasets. The existing backend
remains the default. See the [native augmentation guide](../../../../../vc_reader/docs/augmentation.md)
for the API, target declarations, migration notes, and validation commands.

