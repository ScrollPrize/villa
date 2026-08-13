import torch
import torch.nn as nn
from copy import deepcopy
import logging
import os
os.environ["WANDB_MODE"] = "dryrun" # turn off annoying bugging wandb logging
import sys
# add current path to sys.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything
from datasets.thaumato_dataset import Mask3DInference
from datasets.preprocessing.thaumato_preprocessing import STPLS3DPreprocessing

import pandas as pd
import numpy as np
import concurrent.futures

global initialized
initialized = False

num_gpus = 1

def set_num_gpus(num_gpus_):
    global num_gpus
    num_gpus = num_gpus_

def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        cfg["trainer"][
            "resume_from_checkpoint"
        ] = f"ThaumatoAnakalyptor/mask3d/{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
        
    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers

def init(num_gpus=1):
    set_num_gpus(num_gpus)
    # clear al sys args
    temp_argv = sys.argv
    sys.argv = [sys.argv[0]]  # Keep only the script name, remove all other arguments
    print("Sys ARGV:", sys.argv)
    res = init_()
    sys.argv = temp_argv
    return res

def load_and_merge_configs(base_config_path, config_dir="conf"):
    # Load the base configuration
    base_cfg = OmegaConf.load(base_config_path)

    # Extract the defaults list from the base configuration
    defaults_list = base_cfg.get("defaults", [])

    # Recursive function to merge additional configurations
    def merge_configs(cfg, defaults):
        for default in defaults:
            for key, config_name in default.items():
                # Construct the path to the configuration file
                config_path = os.path.join(config_dir, key, f"{config_name}.yaml")

                # Load the additional configuration
                additional_cfg = OmegaConf.load(config_path)

                # Check if the additional configuration has its own defaults
                if "defaults" in additional_cfg:
                    # Merge the defaults of the additional configuration
                    additional_cfg = merge_configs(additional_cfg, additional_cfg.pop("defaults"))

                # Merge the additional configuration into the base configuration
                cfg[key] = OmegaConf.merge(cfg.get(key, {}), additional_cfg)

        return cfg

    # Merge the additional configurations specified in the defaults list
    cfg = merge_configs(base_cfg, defaults_list)

    return cfg


@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def init_(cfg: DictConfig):
# def init_():
#     base_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf", "config_base_instance_segmentation.yaml")
#     config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf")
#     cfg = load_and_merge_configs(base_config_path, config_dir)
    # print("cfg", cfg, "gpus", num_gpus)
    global models
    global initialized
    if initialized:
        return models
    # Update the configuration with the specific settings
    CURR_DBSCAN = 14.0
    CURR_TOPK = 25
    CURR_QUERY = 160
    CURR_SIZE = 54

    # Manually load the specific config that "stpls3d" points to.
    os.chdir(hydra.utils.get_original_cwd())
    stpls3d_config = OmegaConf.load("ThaumatoAnakalyptor/mask3d/conf/data/datasets/stpls3d_working_inference_but_not_train.yaml")

    OmegaConf.set_struct(cfg, False)
    cfg.general.experiment_name = f'validation_query_{CURR_QUERY}_topk_{CURR_TOPK}_dbscan_{CURR_DBSCAN}_size_{CURR_SIZE}'
    cfg.general.project_name = "stpls3d_test"
    # print("stpl cfg", stpls3d_config)
    cfg.data = OmegaConf.merge(cfg.data, stpls3d_config)
    cfg.general.num_targets = 2
    cfg.data.num_labels = 2
    cfg.data.voxel_size = 0.333
    cfg.data.num_workers = 10
    cfg.data.cache_data = False
    cfg.data.cropping_v1 = False
    cfg.general.reps_per_epoch = 100
    cfg.model.num_queries = CURR_QUERY
    cfg.general.on_crops = False # Initially was True
    cfg.model.config.backbone._target_ = "models.Res16UNet18B"
    cfg.general.train_mode = False
    cfg.general.checkpoint = "ThaumatoAnakalyptor/mask3d/saved/train/last-epoch.ckpt"
    cfg.data.crop_length = CURR_SIZE
    cfg.general.eval_inner_core = 50.0
    cfg.general.topk_per_image = CURR_TOPK
    cfg.general.use_dbscan = False # Initially was on, but uses A LOT of compute on CPU. quite slow.
    cfg.general.dbscan_eps = CURR_DBSCAN
    cfg.data.test_mode = "test"
    cfg.general.export = True
    OmegaConf.set_struct(cfg, True)
    # because hydra wants to change dir for some reason
    models = []
    for gpu_nr in range(num_gpus):
        res = get_parameters(cfg)
        model = res[1]
        model.to(f"cuda:{gpu_nr}")
        #print("dataset config", cfg.data.datasets)
        model.prepare_data()
        # model = nn.DataParallel(model)
        # Add model to list of models
        models.append(model)

    # res = get_parameters(cfg)
    # model_original = res[1]
    # model_original.to(f"cuda:0")
    # global models
    # models = []
    # for gpu_nr in range(1, num_gpus):
    #     model = deepcopy(model_original)
    #     model.to(f"cuda:{gpu_nr}")
    #     #print("dataset config", cfg.data.datasets)
    #     model.prepare_data()
    #     # model = nn.DataParallel(model)
    #     # Add model to list of models
    #     models.append(model)

    # thaumato_preprocessing.py for preprocessed npy loading to npy saving format (containing points object)
    global preprocessing_thaumato
    preprocessing_thaumato = STPLS3DPreprocessing(data_dir='ThaumatoAnakalyptor/mask3d/data/raw/thaumatoanakalyptor', save_dir='ThaumatoAnakalyptor/mask3d/data/processed/thaumatoanakalyptor')
    print("preprocessing_thaumato initialized")
    # thaumato_dataset.py for loading data in pytorch format
    global inference_preprocessing 
    inference_preprocessing = Mask3DInference()
    print("initialized")

    initialized = True
    return models

def get_model():
    if not initialized:
        init()
    return models[0]

# clusters points, detects the papyrus sheet instances
def inference(points_3d, index=0, num_gpus=1):
    '''
    points: np.array of shape (N, 3)

    returns: dict with: "pred_masks", "pred_scores", "pred_classes"
    '''
    
    if not initialized:
        init(num_gpus=num_gpus)

    # Convert labels into processed labels and instance ids.
    processed_labels = np.zeros(points_3d.shape[0])
    instance_ids = np.zeros(points_3d.shape[0])  # Changed -1 to -100 directly
    colors = np.random.rand(*points_3d.shape)

    # Assuming the points array shape is (n, 3) and normals is (n, 3),
    # concatenate them along with labels to form the desired structure.
    # Here, we'll only use the first three columns of normals for now.
    points = np.hstack((points_3d, colors, processed_labels[:, None], instance_ids[:, None]))

    points = preprocessing_thaumato.process_points_thaumato_inference(points, "test")
    item_pytorch = inference_preprocessing.prepare_item(points)
    batch = [item_pytorch]

    global models
    with torch.no_grad():
        predictions = models[index%num_gpus].inference(batch)

    print(predictions.keys())
    print(predictions[0].keys())
    print(predictions[0]["pred_masks"].shape)
    print(predictions[0]["pred_scores"].shape)
    print(predictions[0]["pred_scores"])
    print(predictions[0]["pred_classes"].shape)
    print(predictions[0]["pred_classes"])

    prediction = predictions[0]

    result = np.any(predictions[0]["pred_masks"], axis=1).astype(int)
    print(sum(result))
    print(np.sum(predictions[0]["pred_masks"]))

    return prediction

def batch_inference(points_3d_list, index=0, num_gpus=1):
    '''
    points_3d_list: List of np.array, each of shape (N, 3)

    returns: List of dict, each with: "pred_masks", "pred_scores", "pred_classes"
    '''
    
    if not initialized:
        init(num_gpus=num_gpus)

    global models
    batch = []

    # # Utilizing multithreading for preprocessing
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     preprocessed_pytorch_items = list(executor.map(preprocess_points, points_3d_list))

    # batch.extend(preprocessed_pytorch_items)

    # Single Threaded
    for points in points_3d_list:
        batch.append(preprocess_points(points))

    with torch.no_grad():
        predictions = models[index%num_gpus].inference(batch)

    return predictions

def preprocess_points(points_3d):
    processed_labels = np.zeros(points_3d.shape[0])
    instance_ids = np.zeros(points_3d.shape[0])
    colors = np.random.rand(*points_3d.shape)
    points = np.hstack((points_3d, colors, processed_labels[:, None], instance_ids[:, None]))
    points = preprocessing_thaumato.process_points_thaumato_inference(points, "test")
    item_pytorch = inference_preprocessing.prepare_item(points)
    return item_pytorch

def to_labels(predicitons, filter_class=1):
    '''
    predicitons: dict with: "pred_masks", "pred_scores", "pred_classes"

    returns: np.array of shape (N)
    '''
    # Prediction list of dicts
    if isinstance(predicitons, list):
        labels_list = []
        for prediction in predicitons:
            labels_list.append(to_labels(prediction, filter_class))
        return labels_list

    mask_list = []
    scores_list = []
    # filter out all classes except filter_class
    for i in range(len(list(predicitons["pred_classes"]))):
        if predicitons["pred_classes"][i] == filter_class:
            mask_list.append(predicitons["pred_masks"][:, i])
            scores_list.append(predicitons["pred_scores"][i])

    # construct labels
    labels = np.zeros(mask_list[0].shape)
    for i, mask in enumerate(mask_list):
        mask = mask.astype(bool)
        labels[mask] = i + 1
    
    return labels

def to_surfaces(points, normals, colors, predictions, filter_class=1):
    '''
    points: 3d points, predicitons: dict with: "pred_masks", "pred_scores", "pred_classes"

    returns: np.array of shape (N, 3)
    '''
    # Prediction list of dicts
    if isinstance(predictions, list):
        surfaces_list = []
        surfaces_normals_list = []
        surfaces_colors_list = []
        scores_list_list = []
        for i, prediction in enumerate(predictions):
            surfaces, surfaces_normals, surfaces_colors, scores_list = to_surfaces(points[i], normals[i], colors[i], prediction, filter_class)
            surfaces_list.append(surfaces)
            surfaces_normals_list.append(surfaces_normals)
            surfaces_colors_list.append(surfaces_colors)
            scores_list_list.append(scores_list)
        return surfaces_list, surfaces_normals_list, surfaces_colors_list, scores_list_list
    
    mask_list = []
    scores_list = []
    # filter out all classes except filter_class
    for i in range(len(list(predictions["pred_classes"]))):
        if predictions["pred_classes"][i] == filter_class:
            mask_list.append(predictions["pred_masks"][:, i])
            scores_list.append(predictions["pred_scores"][i])

    # construct surfaces list
    surfaces = []
    surfaces_normals = []
    surfaces_colors = []
    for i, mask in enumerate(mask_list):
        mask = mask.astype(bool)
        surfaces.append(points[mask])
        surfaces_normals.append(normals[mask])
        surfaces_colors.append(colors[mask])

    return surfaces, surfaces_normals, surfaces_colors, scores_list
    

if __name__ == "__main__":
    # initialize for inference once
    init()
    
    # continued another day:
    # "load" raw original data and process into "processed" data. just add dummy values for "gt", will be not looked at anyway during inference and then the whole other things can be use.

    # generate_surface_pointgroup_dataset.py fornpy saving format without actualy saving but as points variable
    points = pd.read_csv('point_clouds/train/point_cloud_27.txt', header=None).values
    print(f"points shape: {points.shape}")

    
    points = preprocessing_thaumato.process_points_thaumato_inference(points, "test")
    print(f"points shape: {points.shape}")

    # thaumato_dataset.py for loading data in pytorch format
    item_pytorch = inference_preprocessing.prepare_item(points)
    print(type(item_pytorch[0]), type(item_pytorch[1]))

    batch = [item_pytorch]

    with torch.no_grad():
        predictions = models[0].inference(batch)

    print(predictions.keys())
    print(predictions[0].keys())
    print(predictions[0]["pred_masks"].shape)
    print(predictions[0]["pred_scores"].shape)
    print(predictions[0]["pred_scores"])
    print(predictions[0]["pred_classes"].shape)
    print(predictions[0]["pred_classes"])

    result = np.any(predictions[0]["pred_masks"], axis=1).astype(int)
    print(sum(result))
    print(np.sum(predictions[0]["pred_masks"]))
    

    # dict with: "pred_masks", "pred_scores", "pred_classes"

    