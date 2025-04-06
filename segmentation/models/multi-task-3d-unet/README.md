this repository contains training and inference scripts for a dynamically created 3d u-net with residual encoders/decoders with squeeze and excitation blocks. 

the model can learn both segmentation and regression tasks simultaneously, and can accept an arbitrary number of inputs and target labels. currently the only supported data format is zarr. 

currently all of the 3d blocks from pytorch3dunet are implemented, but adding additional support is trivial and something i will do soon. currently the model only supports a shared encoder path with separate decoder heads/paths. 

i'm adding to this frequently, so more will come. the design im going for is very much inspired by nnunet (it would probably make more sense to just extend nnunet but wheres the fun in that). it borrows heavily from pytorch3dunet, and all of the building blocks are from there. i am very grateful to both of these development teams/individuals for sharing and developing these frameworks :) 

https://github.com/MIC-DKFZ/nnUNet

https://github.com/wolny/pytorch-3dunet

![image](https://github.com/user-attachments/assets/08f27dea-5b93-4b4d-a97f-b53bb6921cf3)


___
## purpose
the primary focus of creating this was to enable efficient multi-task learning with 3d unets. i am not a particularly skilled or experience programmer nor am i a machine learning expert by any stretch of the imagination, no guarantees for performance!

### design
the repository is setup with a focus on modularity -- the two scripts train.py and inference.py are the entry points, and these are configured through the use of a configuration manager class (aptly named ConfigManager) which parses the configuration yaml and assigns the items to their respective fields.

each step of the training configuration is handled by methods of the BaseTrainer class -- the model train script runs in this order:

1. The script is started with `python train.py --config_path ./tasks/task.yaml` . 

2. The `ConfigManager` class is instantiated with this config file path, and parses the file, assigning the contents to the appropriate properies, and using reasonable defaults when none are provided.
 
2. `_build_model` creates a model from the configurations provided by the ConfigManager, and prints its configuration

3. `_configure_dataset` receives the patch size, task list, label ratios, and other arguments from the ConfigManager and creates a Zarr dataset. 
   - It searches through a chosen reference zarr for regions of patch size that contain some parameters for label volume and density, and assigns these to valid patches 
   - These valid patches are gathered from the indices and passed through to `__getitem__`
   -  Some augmentations are performed and the data is converted to torch tensors with shape (c, z, y, x), dtype of float32, and values between 0 and 1
   -  This data now in pytorch compatible format is returned to the training script

7. `_build_loss` receives the loss classes from ConfigManager, finds it among the string to class mapping defined in the function, and assigns each loss to each task

8. `_get_optimizer` receives the optimizer from the ConfigManager, and sets it as the optimizer for this run

9. `_get_scheduler` receives the scheduler class from the ConfigManager, and assigns the correct class from the mapping defined in the function

10. `_configure_dataloaders` receives the batch size and num_workers from the ConfigurationManager and instantiates the training and validation dataloaders

11. Gradient accumulation steps are sent from the ConfigManager

12. If a checkpoint is provided, the weights are loaded along with the optimizer, scheduler state, and epoch number. If weights only is set by the ConfigManager, only the weights are loaded and training is begun at epoch 0 with a fresh optimizer and scheduler.

11. The training loop is started:
    
    - For each item in `data_dict` (this is a dictionary returned by the dataset that contains all images and labels):
    - If it's the first batch, the script prints off the shape, dtype, and min/max values contained
    - The item named 'image' in the data_dict is sent to the device
    - Each other item in the `data_dict`, which we assume are all labels, are sent to the device -- these are stored in the `targets_dict`, by key(name) and item(data)
    - The outputs of the model are received
    - For each item in the targets_dict, the loss is computed
    - The gradients are sent back 
    - The weights are updated
    - The checkpoint is saved

19. Validation is performed, following the same steps save updating the weights or gradients. Loss is still computed for metrics. 

20. A gif is saved in the directory set by the ConfigManager containing the raw data, and each targets label/prediction pair

21. The next epoch is started 

Outside the training loop itself, each part is easily extendable or replaceable by subclassing the appropriate method. If you wanted to use different losses, you could add them to the mapping, or simply create a new training script, subclass the BaseTrainer with something like DifferentLossTrainer and then define only `_get_loss`, replacing the current method with whatever you want. So long as your loss can accept tensors of shape `b, c, z, y, x` , you have nothing else to do. 
___

## configuration
the training and validation are configured through a class called `ConfigManager`. This class parses a yaml file, an example of which is in the tasks folder. each of the following properties are set if provided, with some defaults selected if they are not contained in the config file. if you want to add a configuration, just put it in the yaml, and assign the property here-- you can now access it with the ConfigManager anywhere in training, inference, or within the dataset.

not all of properties defined here are currently in use -- this is a fun side project for me that i am actively working on. 
```python

from types import SimpleNamespace
from pathlib import Path
import yaml

class ConfigManager:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.tr_info = SimpleNamespace(**config["tr_setup"])
        self.tr_configs = SimpleNamespace(**config["tr_config"])
        self.model_config = SimpleNamespace(**config["model_config"])
        self.dataset_config = SimpleNamespace(**config["dataset_config"])
        self.inference_config = SimpleNamespace(**config["inference_config"])

        # training setup
        self.model_name = getattr(self.tr_info, "model_name", "Model")
        self.vram_max = float(getattr(self.tr_info, "vram_max", 16))
        self.autoconfigure = bool(getattr(self.tr_info, "autoconfigure", True))
        self.tr_val_split = float(getattr(self.tr_info, "tr_val_split", 0.95))
        self.dilate_label = bool(getattr(self.tr_info, "dilate_label", False))
        self.ckpt_out_base = Path(getattr(self.tr_info, "ckpt_out_base", "./checkpoints/"))
        self.checkpoint_path = getattr(self.tr_info, "checkpoint_path", None)
        self.load_weights_only = getattr(self.tr_info, "load_weights_only", False)
        self.tensorboard_log_dir = str(getattr(self.tr_info, "tensorboard_log_dir", "./tensorboard_logs/"))

        # parameters for training
        self.loss_only_on_label = bool(getattr(self.tr_configs, "loss_only_on_label", False))
        self.train_patch_size = tuple(getattr(self.tr_configs, "patch_size", [192, 192, 192]))
        self.train_batch_size = int(getattr(self.tr_configs, "batch_size", 2))
        self.gradient_accumulation = int(getattr(self.tr_configs, "gradient_accumulation", 1))
        self.optimizer = str(getattr(self.tr_configs, "optimizer", "AdamW"))
        self.ignore_label = getattr(self.tr_configs, "ignore_label", None)
        self.max_steps_per_epoch = int(getattr(self.tr_configs, "max_steps_per_epoch", 500))
        self.max_val_steps_per_epoch = int(getattr(self.tr_configs, "max_val_steps_per_epoch", 25))
        self.train_num_dataloader_workers = int(getattr(self.tr_configs, "num_dataloader_workers", 4))
        self.label_smoothing = float(getattr(self.tr_configs, "label_smoothing", 0.2))
        self.max_epoch = int(getattr(self.tr_configs, "max_epoch", 500))
        self.initial_lr = float(getattr(self.tr_configs, "initial_lr", 1e-3))
        self.weight_decay = float(getattr(self.tr_configs, "weight_decay", 0))
        self.tensorboard_log_dir = Path(self.tensorboard_log_dir) / self.model_name

        # model configuration -- no defaults here because it's handled by build_network_from_config dynamically
        self.model_kwargs = vars(self.model_config).copy()

        # dataset config
        self.min_labeled_ratio = float(getattr(self.dataset_config, "min_labeled_ratio", 0.1))
        self.min_bbox_percent = float(getattr(self.dataset_config, "min_bbox_percent", 0.95))
        self.use_cache = bool(getattr(self.dataset_config, "use_cache", True))
        self.cache_file = Path((getattr(self.dataset_config, "cache_file", 'valid_patches.json')))
        self.in_channels = int(getattr(self.dataset_config, "in_channels", 1))
        self.tasks = self.dataset_config.targets
        self.volume_paths = self.dataset_config.volume_paths
        self.out_channels = ()
        for task_name, task_info in self.tasks.items():
            self.out_channels += (task_info["channels"],)

        # inference config
        self.infer_input_path = str(getattr(self.inference_config, "input_path", None))
        self.infer_input_format = str(getattr(self.inference_config, "input_format", "zarr"))
        self.infer_output_format = str(getattr(self.inference_config, "output_format", "zarr"))
        self.infer_load_all = bool(getattr(self.inference_config, "load_all", False))
        self.infer_output_dtype = str(getattr(self.inference_config, "output_type", "np.uint8"))
        self.infer_output_targets = list(getattr(self.inference_config, "output_targets", "all"))
        self.infer_overlap = float(getattr(self.inference_config, "overlap", 0.15))
        self.infer_blending = str(getattr(self.inference_config, "blending", "gaussian_importance"))
        self.infer_patch_size = tuple(getattr(self.inference_config, "patch_size", self.train_patch_size))
        self.infer_batch_size = int(getattr(self.inference_config, "batch_size", self.train_batch_size))
        self.infer_checkpoint_path = getattr(self.inference_config, "checkpoint_path", None)
        self.load_strict = bool(getattr(self.inference_config, "load_strict", True))
        self.infer_num_dataloader_workers = int(getattr(self.inference_config, "num_dataloader_workers", self.train_num_dataloader_workers))

        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        else:
            self.checkpoint_path = None

```
