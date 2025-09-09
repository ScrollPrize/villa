# VC-Surface-Models

All training data and model weights will be located here: https://dl.ash2txt.org/community-uploads/bruniss/p2-submission. Most training was done with a skeletonization based loss from MIC-DKFZ located here https://github.com/MIC-DKFZ/Skeleton-Recall. Some additional training was done using a distance transform weighted loss, with some info here: https://github.com/MIC-DKFZ/nnUNet/pull/2630

Heavy additional augmentations during training in the form of elastic deformation, an inhomogeneous illumination transform, and blank rectangle transforms inspired by this repo : https://github.com/MIC-DKFZ/MurineAirwaySegmentation (if you cant tell yet i'm a huge fan of everything MIC-DKFZ does). 

The final train dataset is comprised of 561 volumes averaging around 215^3 , half of which come from p.herc 1667 and half of which come from the grand prize region of p.herc paris 1. In all , the total volume is less than 4% of the GP region. 

nnUNet has fantastic documentation if you run into any issues along the way: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md

Install miniconda:

`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh `

`bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3 `

`source $HOME/miniconda3/bin/activate`

Install torch:

`conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`

 
Install batchgeneratorsv2"

`https://github.com/bruniss/batchgeneratorsv2.git 
cd batchgeneratorsv2
pip install -e .`

Install nnunetv2
`git clone https://github.com/bruniss/VC-Surface-Models.git
cd VC-Surface-Models
pip install -e .`

### inference
`nnUNetv2_predict -i "/workspace/in" -o "/workspace/out" -d 036 -c 3d_fullres -f 0 -p nnUNetResEncUNetMPlans --save_probabilities`
`nnUNetv2_predict -i "/workspace/in" -o "/workspace/out" -d 043 -c 3d_fullres -f 0 -p SkeletonRecall__nnUNetResEncUNetPlans_41G --save_probabilities`
`nnUNetv2_predict -i "/workspace/in" -o "/workspace/out" -d 044 -c 3d_fullres -f 0 -p SkeletonRecall__nnUNetResEncUNetPlans_41G --save_probabilities`
`nnUNetv2_predict -i "/workspace/in" -o "/workspace/out" -d 050 -c 3d_fullres -f 0 -p nnUNetTrainerDistDiceCELossExtraDA__nnUNetResEncUNetMPlans --save_probabilities`

Then ensemble these predictions together, changing paths in script:
`python adaptive_merge.py`

The ensemble used for our final trace was created by taking the softmax predictions of 4 models. For each slice of each volume tif, a 9x9 sliding window computes the variance within it, and then weighs them from highest to lowest variance for each window. The slices from each are then blended, and the next slice is computed. The reason for this custom ensembling rather than the default nnUNet mean ensembling is because a simple arithmetic mean for each prediction results in over confident models dominating the ensemble, even if their respective predictions are poor. In this method a model that predicts a solid white block is penalized accordingly, because its prediction for our region is poor. Most of the time a large segmented region is desired, but for fine and detailed structures, this results in bad models ruining the ensemble. The resultant volumes from the two methods are very similar, but in regions where it counts most, this led to better results for us. 

Threshold these using otsu's method: 
`python apply_thresholds.py`

Combine to a zarr
`python grids_to_zarr.py`

Convert to an ome-zarr (thanks chuck! more info here https://github.com/KhartesViewer/scroll2zarr/blob/main/README.md#user-guide-for-zarr_to_ome):
`python zarr_to_ome.py /path/input.zarr /path/output.zarr` 

### train 
Preprocess: `nnUNetv2_plan_and_preprocess -d 050 -pl nnUNetPlannerResEncL -c 3d_fullres --verify_dataset_integrity`
Run training: `nnUNetv2_train 050 3d_fullres 0 -p nnUNetResEncUNetMPlans`
