an attempt at a faithful implementation of dinov2-style pretraining on 3d volumes. 

- the dinov2_eva is from [dynamic-network-architectures](github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/architectures/dinov2_eva.py) 
- the augmentation library is a loosely modified [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2)
- normalization is mostly borrowed from [nnunetv2](https://github.com/MIC-DKFZ/nnUNet)

this implementation is still incomplete. pretraining works but no finetuning yet written. 