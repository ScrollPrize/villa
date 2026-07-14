# Giorgio Angelotti - 2024

# Train/Fine Tune SAM 2 on scrolls for photogrammetry

# this file is a readaptation of this: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/8b6d59d8f764a2a1d9f018ef949571ddcac57c9b/TRAIN.py

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import wandb
import os

# Define augmentation pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.2),  # Randomly flip images horizontally
    A.VerticalFlip(p=0.2),  # Randomly flip images vertically
    A.RandomRotate90(p=0.2),  # Rotate 90 degrees randomly
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.1),  # Shift, Scale, Rotate
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),  # Change brightness/contrast
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # Apply Gaussian Blur
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),  # Elastic distortions
    A.Resize(1024, 1024),  # Resize to fixed 1024x1024
])

# Path to the main dataset folder
data_dir = r"photo_data/"

# Collect all image-mask pairs (for JPG and JPG_Enhanced)
data = []  # list to store dataset samples
for scroll_folder in os.listdir(data_dir):
    scroll_path = os.path.join(data_dir, scroll_folder)
    if os.path.isdir(scroll_path):  # Check if it's a directory
        for orientation_folder in os.listdir(scroll_path):
            orientation_path = os.path.join(scroll_path, orientation_folder)
            if os.path.isdir(orientation_path):
                # Get JPG-Mask pairs
                jpg_path = os.path.join(orientation_path, "JPG")
                mask_path = os.path.join(orientation_path, "Masks")
                if os.path.exists(jpg_path) and os.path.exists(mask_path):
                    for filename in os.listdir(jpg_path):
                        if filename.lower().endswith(".JPG"):
                            mask_filename = filename[:-4] + "_mask.png"
                            if os.path.exists(os.path.join(mask_path, mask_filename)):
                                data.append({
                                    "image": os.path.join(jpg_path, filename),
                                    "annotation": os.path.join(mask_path, mask_filename)
                                })
                
                # Get JPG_Enhanced-Mask pairs
                jpg_enhanced_path = os.path.join(orientation_path, "JPG_Enhanced")
                if os.path.exists(jpg_enhanced_path) and os.path.exists(mask_path):
                    for filename in os.listdir(jpg_enhanced_path):
                        if filename.lower().endswith(".jpg"):
                            mask_filename = filename[:-4] + "_mask.png"
                            if os.path.exists(os.path.join(mask_path, mask_filename)):
                                data.append({
                                    "image": os.path.join(jpg_enhanced_path, filename),
                                    "annotation": os.path.join(mask_path, mask_filename)
                                })

# Function to read and augment a random batch from the dataset
def read_batch(data, training=True):
    ent = data[np.random.randint(len(data))]  # Randomly select an entry
    Img = cv2.imread(ent["image"])[..., ::-1]  # Read image in RGB
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read mask

    # Apply augmentation
    if training:
        augmented = augmentation(image=Img, mask=ann_map)
        Img = augmented["image"]
        ann_map = augmented["mask"]

    # Get binary masks and points
    inds = np.unique(ann_map)[1:]  # Find unique labels (skip background)
    points = []
    masks = []
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([[yx[1], yx[0]]])

    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])


# Load model
model_type = "t"
model_type_long = "tiny"


sam2_checkpoint = f"./checkpoints/sam2.1_hiera_{model_type_long}.pt" # "sam2_hiera_large.pt"
model_cfg = f"sam2_hiera_{model_type}.yaml" # "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler() # mixed precision

# Create WandB config using existing variables
config = {
    "data_dir": data_dir,
    "model": {
        "type": model_type,
        "type_long": model_type_long,
        "checkpoint": sam2_checkpoint,
        "config_file": model_cfg
    },
    "training": {
        "loss_ewa_weight": 0.01,
        "num_iterations": 100000,
        "score_loss_weight": 0.05,
        "learning_rate": optimizer.defaults['lr'],
        "weight_decay": optimizer.defaults['weight_decay'],
        "optimizer": type(optimizer).__name__,
        "mixed_precision": isinstance(scaler, torch.cuda.amp.GradScaler)
    },
    "resize_max_dim": 1024,  # Maximum dimension for resizing images
    "device": "cuda"
}

# Training loop
wandb_run = wandb.init(project="sam2-photogrammetry", config=config)

for itr in range(100000):
    with torch.cuda.amp.autocast(): # cast to mix precision
        #with torch.cuda.amp.autocast():
            image,mask,input_point, input_label = read_batch(data) # load data batch
            if mask.shape[0]==0: continue # ignore empty batches
            predictor.set_image(image) # apply SAM image encodet to the image

            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            # mask decoder

            batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            # Usage inside your training loop:
            #seg_loss = focal_loss(prd_mask, gt_mask, alpha=0.25, gamma=2.0)

            # Score loss calculation (intersection over union) IOU

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss=seg_loss+score_loss*0.05  # mix losses

            # apply back propogation

            predictor.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision

            if itr%1000==0: torch.save(predictor.model.state_dict(), f"photo2_{model_type}_{itr}.torch") # save model

            # Display results

            if itr==0: mean_iou=0
            mean_iou_iter = np.mean(iou.cpu().detach().numpy())
            mean_iou = mean_iou * 0.99 + 0.01 * mean_iou_iter
            print("step)",itr, "Accuracy(IOU)=",mean_iou, " Iteration(IOU)",mean_iou_iter)
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], "iteration_iou":mean_iou_iter, "score_loss":score_loss, "seg_loss":seg_loss, "train_loss": loss, "train_iou": mean_iou}, step=itr)