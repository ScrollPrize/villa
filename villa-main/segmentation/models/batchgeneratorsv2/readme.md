# batchgeneratorsv2
This repository is work in progress. If builds upon the [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) 
framework but makes several key changes to the transforms:

1. Transforms now explicitly distinguish between data types: images, segmentation, pixel-wise regression target, keypoints, bbox
2. All transforms have been reimplemented from scratch with a focus on performance. In case of performance parity 
between previous numpy and new torch-based implementations, preference is given to pytorch. 
3. Transforms are applied on a sample level, not a batch level as was done previously!

Caveats:
- performance is optimized for CPU. GPU-based data augmentation is not supported (implementation may use numpy etc) and will not be supported
- currently this repository only covers a small subset of the transforms available in batchgenerators. Feel free to contribute more

### How to contribute
We are happy to accept PRs that further optimize performance and extend the available transformations!

- Please provide benchmarking results relative to the old batchgenerators implementation (if applicable)
- Please stick to the current transform template!

# Acknowledgements
<img src="assets/HI_Logo.png" height="100px" />

<img src="assets/dkfz_logo.png" height="100px" />

batchgeneratorsv2 developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
