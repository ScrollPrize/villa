# vesuvius
From [Vesuvius Challenge](https://scrollprize.org), a Python library for : 
- Accessing CT data of Herculaneum Scrolls
- Training 2d or 3d semantic segmentation models, with support for multi-task and multi-class
- Training 2d or 3d models on regression tasks
- Inferring with models trained with the trainers provided in the package or with pretrained nnUNetv2 models
  - Inference can be performed on remote data (http, s3) stored as Zarr arrays
- Rendering .obj segments with local or remote data
- Voxelizing large .obj segmentations for use as 3d labels
- Preprocessing labels of fiber-like structures
- Interactive labeling and model training through a Napari based trainer
- Proofreading large arrays of image/label pairs and saving approved chunks
- Computing structure tensors on large Zarr arrays, and deriving eigenvalues and eigenvectors from them

`vesuvius` also comes prepackaged with: 
- extensive data augmentation
- PyTorch datasets for zarr or image (png, jpg, tif) data

_this package is in active development_


### Entrypoints: 

| Name                          | Script                             | Description                                                                                                                                                                                               |
| ----------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vesuvius.predict`            | `models.run.inference`             | outputs logits from a pretrained nnUNet-v2 model or one trained within the Vesuvius training framework; currently only works on Zarr data and can be fully distributed with `--num_parts` and `--part_id` |
| `vesuvius.blend_logits`       | `models.run.blending`              | blends the logits from `vesuvius.predict` using Gaussian blending                                                                                                                                         |
| `vesuvius.finalize_outputs`   | `models.run.finalize_outputs`      | performs softmax / argmax / none on the blended array and writes a final `uint8` volume                                                                                                                   |
| `vesuvius.inference_pipeline` | `models.run.vesuvius_pipeline`     | runs the three steps above (predict → blend → finalize) in sequence                                                                                                                                       |
| `vesuvius.compute_st`         | `structure_tensor.run_create_st`   | computes structure tensors on input data and derives eigen-values/vectors                                                                                                                                 |
| `vesuvius.napari_trainer`     | `utils.napari_trainer.main_window` | launches a Napari window for interactive training and inference                                                                                                                                           |
| `vesuvius.proofreader`        | `utils.vc_proofreader.main`        | opens a Napari window that loads local / remote image-label arrays and extracts training patches                                                                                                          |
| `vesuvius.voxelize_obj`       | `scripts.voxelize_objs`            | converts input `.obj` meshes to voxel grids and outputs `.tif` stacks                                                                                                                                     |
| `vesuvius.refine_labels`      | `scripts.edt_frangi_label`         | refines surface or fibre labels with a custom Frangi-based filter                                                                                                                                         |
| `vesuvius.render_obj`         | `rendering.mesh_to_surface`        | renders `.obj` meshes and outputs their surface-volume layers                                                                                                                                             |
| `vesuvius.flatten_obj`        | `rendering.slim_uv`                | flattens an `.obj` mesh using `slim_uv`                                                                                                                                                                   |
| `vesuvius.train`              | `models.run.train`                 | main entry point for training models                                                                                                                                                                      |




## 📓 Introductory notebooks
To get started, we recommend these notebooks that jump right in:

1. 📊 [Scroll Data Access](https://colab.research.google.com/github/ScrollPrize/vesuvius/blob/main/notebooks/example1_data_access.ipynb): an introduction to accessing scroll data using a few lines of Python!

2. ✒️ [Ink Detection](https://colab.research.google.com/github/ScrollPrize/vesuvius/blob/main/notebooks/example2_ink_detection.ipynb): load and visualize segments with ink labels, and train models to detect ink in CT.

3. 🧩 [Volumetric instance segmentation cubes](https://colab.research.google.com/github/ScrollPrize/vesuvius/blob/main/notebooks/example3_cubes_bootstrap.ipynb): how to access instance-annotated cubes with the `Cube` class, used for volumetric segmentation approaches.

## `vesuvius` does:
- **Data retrieval**: Fetches volumetric scroll data, surface volumes of scroll segments, and annotated volumetric instance segmentation labels. Remote repositories and local files are supported.
- **Data listing**: Lists the available data on [our data server](https://dl.ash2txt.org).
- **Data caching**: Caches fetched data to improve performance when accessing remote repositories.
- **Normalization**: Provides options to normalize data values.
- **Multiresolution**: Accesses and manages data at multiple image resolutions.

## `vesuvius` doesn't do:
- **Remote data modification**: The read-only library does not support modifying the original data.
- **Complex analysis**: While it provides access to data, it does not include built-in tools for complex data analysis or visualization.

## Installation

`vesuvius` can be installed with `pip`.
Then, before using the library for the first time, accept the license terms:
```sh
$ pip install vesuvius
$ vesuvius.accept_terms --yes
```

___
**Note:** 

The model framework utilized by `vesuvius` is _heavily_ inspired by [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet). The default configuration will use the same blocks and construct the same encoders/decoders as the default nnUNetv2 ResEncUNet. As such, a significant portion of the code from the nnUNetv2 repository is duplicated here, with modifications to enable multi-task support with dynamic decoder branches, as well as other data formats. 

Additionally, the augmentations provided within this package are from another of [MIC-DKFZ's](https://github.com/MIC-DKFZ) fantastic array of machine learning libraries, in this case [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2). 

Copying the modules directly into `vesuvius` was a choice of end-user friendliness, as we were using highly modified branches of both libraries, which created conflicts if an end user were to attempt to run any of our models.

_**Detailed documentation for training and inference are located in [the docs folder](/docs)**_
___


### Napari based training

1. Run `vesuvius.napari_trainer`
2. Add an image to the viewer
3. Add a label layer, with the suffix as the name of your target (ex : `32_ink`)
4. Optionally, add a label layer with the suffix `_mask` (ex : `32_mask`)
5. Set your training configuration and hit `run training`
6. Infer on the same layer, or another by importing an image, selecting it in the inference widget, and hitting `Run Inference`


![alt text](docs/images/napari_trainer.png)

___

### Proofreading labels

1. Update the config in `/utils/vc_proofreader/config.py` with the proper paths
2. Run `vesuvius.proofreader`
3. Select your desired patch size and min labeled percentage 
4. Click `run` 
5. Approve patches with `a` or by checking the box
6. Skip patches or continue with `spacebar` or `next pair`
7. Patches are saved in the output dir, and their locations in the .json progress file

![alt text](docs/images/proofreader.png)

### Training with `vesuvius.train`

Supported model types:
- Single task, single class semantic segmentation / regression
- Single task, multi-class semantic segmentation / regression
- Multi-task, single-class semantic segmentation / regression
- Multi-task, multi-class semantic segmentation / regression

By default, when provided with a single channel (binary) input label, the model will output 2 channels (fg/bg), but can adapt to any number of input channels.

**Place your data in the following format:** 
```
data/
  images/
    volume1.zarr (or .tif, .png, .jpg)
    volume2.zarr
  labels/
    volume1_ink.zarr (or tif, .png, .jpg) 
    volume1_hz_fiber.zarr ( if you want an additional task for the same volume )
    volume2_ink.zarr
   ```
**Begin training with:**
```bash
 vesuvius.train -i /path/to/data --batch-size 4 --patch-size 128,128,128 --model-name /path/to/save/checkpoints --config-path /vesuvius/models/configuration/multi-task_config.yaml
```
There are a number of other optional parameters, which you can find with `--help`

When this command is run: 
- a ConfigManager class will be instantiated, which will take the arguments given and the configuration file, and store these as properties.
- The trainer class will execute its class method `__build_model`, which will create an instance of `NetworkFromConfig`
- `NetworkFromConfig` will dynamically determine the number of pooling operations, stages, feature map sizes, operation dimensionality, and other specified parameters
- The trainer class will execute its `_configure_dataset` method, and create an instance of the proper Pytorch dataset
- The trainer class will execute the rest of the setup required for training, through the following additional class methods:
  - `_build_loss`
  - `_get_optimizer`
  - `_get_scheduler`
  - `_get_scaler`
  - `_configure_dataloaders`

- The training loop will begin

Training will output the current losses in the `tqdm` progress bar, and will save a "debug" png or gif (depending on dimensionality) in the checkpoint directory. 

By default, the last 10 checkpoints are saved. This is not a smart way to do it, and will be changed to just store the last 3 + last 2 best validation. 

Training will run for 1,000 epochs by default, with 200 batches/epoch. This can be modified through the configuration file, which can be optionally provided to `vesuvius.train`, and some examples are provided in the [models folder](models/configuration/)

### Running inference with `vesuvius.inference_pipeline`

Detailed documentation is available in [the inference readme](docs/inference.md)

The inference script used here is unfortunately (by necessity) quite convoluted. It **stores a very large amount of intermediate data** , in the form of float16 logits from patches. In the case that you do not have a lot of storage space, it might make sense to borrow some of the functions from the inferer, and rewrite the inference process to write directly to uint8 final arrays.

The inference process is in 3 parts -- inference, blending, and finalization

1. `inference.py` creates a vc_dataset, which reads data from the Volume class in volume.py
 
    - patches are ran through forward passes in batches, and either rotation or mirroring TTA is applied
    - patches are stored in intermediate 'logits' arrays, named by their gpu rank, in a zarr array of shape `patch_z, patch_y, patch_x * num_patches`. It's just a zarr array that is very long and skinny. the reason for this is that writing to overlapping chunks in zarr arrays (which we have because we take 50% of the patch size as overlap during inference) is difficult when you have to keep pace with your gpus to avoid unbounded memory growth. We went with a map-reduce style here to avoid this entirely.
2. the logits arrays are read by `blending.py` chunkwise, along with their 8 neighbors (which contribute to their final values), and accumulated into a "weights array"
   - this accumulation array is normalized chunkwise by the number of patches which contributed to its values
3. the blended array is read by `finalize_outputs.py` , which applies the activation functions (softmax, argmax, or none), casts to uint8, and writes the final zarr

This pipeline can be performed individually, through `vesuvius.predict` , `vesuvius.blend_logits` and `vesuvius.finalize_outputs`, respectively. It also can be performed sequentially with `vesuvius.inference_pipeline` 

On a single machine, running the full pipeline is recommended. You can use local volumes or remote (s3, http) volumes. To begin inference, run
```
vesuvius.inference_pipeline --input /path/to/input.zarr --output /path/to/output.zarr --model hf://scrollprize/surface_recto --batch-size 4
```
You can use local models or models hosted on huggingface hub, so long as they are either nnUNetv2 models or models created with the vesuvius trainer. We have some of our fiber and surface models hosted [here](https://huggingface.co/collections/scrollprize/representation-67e1b44299d5c18f5845874f), but you could use your own as well. 

To use a local model point at either the `nnUNet_results` folder which contains the `dataset.json` and the `fold` directories, or if using a vesuvius model, the `checkpoint.pth` file

### Rendering and Flattening objs
Documentation is provided in [the rendering folder](rendering/README.md)

