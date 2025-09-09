import os
import subprocess
from tap import Tap
import glob

class InferenceArgumentParser(Tap):
    segment_id: list[str] =[]           # Leave empty to process all segments in the segment_path
    segment_path:str='./eval_scrolls'
    model_path:str= 'outputs/vesuvius/pretraining_all/vesuvius-models/valid_20230827161847_0_fr_i3depoch=7.ckpt'
    out_path:str=""
    stride: int = 32
    start_idx:int=17
    workers: int = 4
    batch_size: int = 64
    size:int=64
    reverse:int=0
    device:str='cuda'
    gpus:int=1
    model_compile:bool=True
args = InferenceArgumentParser().parse_args()

# Generate a string "0,1,2,...,args.gpus-1"
gpu_ids = ",".join(str(i) for i in range(args.gpus))

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
import torch
from warmup_scheduler import GradualWarmupScheduler
import wandb
import random
import gc
import pytorch_lightning as pl
import scipy.stats as st
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
print(f"Using {torch.cuda.device_count()} GPUs")

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class CFG:

    # ============== model cfg =============
    in_chans = 26 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 64
    tile_size = 64
    seed = 42
    # ============== augmentation =============
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

def cfg_init(cfg, mode='val'):
    set_seed(cfg.seed)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image_mask(fragment_id,start_idx=18,end_idx=38,rotation=0):
    images = []
    idxs = range(start_idx, end_idx)

    for i in idxs:
        fragment_path = f"{args.segment_path}/{fragment_id}/layers/{i:02}"
        if os.path.exists(f"{fragment_path}.tif"):
            image = cv2.imread(f"{fragment_path}.tif", 0)
        else:
            image = cv2.imread(f"{fragment_path}.jpg", 0)
        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image=np.clip(image,0,200)
        images.append(image)
    images = np.stack(images, axis=2)
    if any(id_ in fragment_id for id_ in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']):
        print("Reverse Segment")
        images=images[:,:,::-1]
    fragment_mask=None
    wildcard_path_mask = f'{args.segment_path}/{fragment_id}/*_mask.png'
    if os.path.exists(f'{args.segment_path}/{fragment_id}/{fragment_id}_mask.png'):
        fragment_mask=cv2.imread(f"{args.segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    elif len(glob.glob(wildcard_path_mask)) > 0:
        # any *mask.png exists
        mask_path = glob.glob(wildcard_path_mask)[0]
        fragment_mask = cv2.imread(mask_path, 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        # White mask
        fragment_mask = np.ones_like(images[:,:,0]) * 255

    return images, fragment_mask

def get_img_splits(fragment_id,s,e,rotation=0):
    images = []
    xyxys = []
    if not os.path.exists(f"{args.segment_path}/{fragment_id}"):
        fragment_id = fragment_id + "_superseded"
    print('reading ',fragment_id)
    # check for superseded fragment
    try:
        image,fragment_mask = read_image_mask(fragment_id, s,e,rotation)
    except Exception as e:
        print("aborted reading fragment", fragment_id, e)
        return None

    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, args.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, args.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])
    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers, pin_memory=(args.gpus==1), drop_last=False,
                              )
    return test_loader, np.stack(xyxys),(image.shape[0],image.shape[1]),fragment_mask

def get_transforms(data, cfg):
    if data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        return image,xy
    
class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=64,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        self.backbone=TimeSformer(
                dim = 512,
                image_size = 64,
                patch_size = 16,
                num_frames = 30,
                num_classes = 16,
                channels=1,
                depth = 8,
                heads = 6,
                dim_head =  64,
                attn_dropout = 0.1,
                ff_dropout = 0.1
            )
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        x = self.backbone(torch.permute(x, (0, 2, 1,3,4)))
        x=x.view(-1,1,4,4)        
        return x


def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    mask_count_kernel = np.ones((CFG.size, CFG.size))
    kernel = gkern(CFG.size, 1)
    kernel = kernel / kernel.max()
    model.eval()

    kernel_tensor = torch.tensor(kernel, device=device)  # Move the kernel to the GPU

    for step, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
        y_preds = torch.sigmoid(y_preds)  # Keep predictions on GPU

        # Resize all predictions at once
        y_preds_resized = F.interpolate(y_preds.float(), scale_factor=16, mode='bilinear')  # Shape (batch_size, 1, 64, 64)
        
        # Multiply by the kernel tensor
        y_preds_multiplied = y_preds_resized * kernel_tensor  # Broadcasting kernel to all images in the batch
        y_preds_multiplied = y_preds_multiplied.squeeze(1)
        # Move results to CPU as a NumPy array
        y_preds_multiplied_cpu = y_preds_multiplied.cpu().numpy()  # Shape: (batch_size, 64, 64)

        # Update mask_pred and mask_count in a batch manner
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += y_preds_multiplied_cpu[i]
            mask_count[y1:y2, x1:x2] += mask_count_kernel

    mask_pred /= np.clip(mask_count, a_min=1, a_max=None)
    return mask_pred
import gc

if __name__ == "__main__":
    # Loading the model
    try:
        model = RegressionPLModel.load_from_checkpoint(args.model_path, strict=False)
    except:
        model = RegressionPLModel(pred_shape=(1,1))
        w=torch.load(args.model_path,weights_only=False)
        model.load_state_dict(w['state_dict'])
    
    if args.model_compile:
        model=torch.compile(model)
    if args.gpus > 1:
        model = DataParallel(model)  # Wrap model with DataParallel for multi-GPU
    model.to(device)
    model.eval()
    wandb.init(
        project="Vesuvius", 
        name=f"ALL_scrolls_tta", 
        )

    # Set up segments
    if len(args.segment_id) == 0:
        args.segment_id = [os.path.basename(x) for x in glob.glob(f"{args.segment_path}/*") if os.path.isdir(x)]
        # Sort the segment IDs
        args.segment_id.sort()
        print(f"Found {len(args.segment_id)} segments: {args.segment_id}")

    try:
        for fragment_id in args.segment_id:
            preds=[]
            try:
                for r in [0]:
                    for i in [args.start_idx]:
                        start_f=i
                        end_f=start_f+CFG.in_chans
                        img_split = get_img_splits(fragment_id,start_f,end_f,r)
                        if img_split is None:
                            continue
                        test_loader,test_xyxz,test_shape,fragment_mask = img_split
                        mask_pred = predict_fn(test_loader, model, device, test_xyxz,test_shape)
                        mask_pred = np.clip(np.nan_to_num(mask_pred),a_min=0,a_max=1)
                        mask_pred /= mask_pred.max()

                        preds.append(mask_pred)

                        if len(args.out_path) > 0:
                            # CV2 image
                            image_cv = (mask_pred * 255).astype(np.uint8)
                            try:
                                os.makedirs(args.out_path,exist_ok=True)
                            except:
                                pass
                            cv2.imwrite(os.path.join(args.out_path, f"{fragment_id}_prediction_rotated_{r}_layer_{i}.png"), image_cv)
                        del mask_pred
                if len(preds) > 0:
                    img=wandb.Image(
                    preds[0], 
                    caption=f"{fragment_id}"
                    )
                    wandb.log({'predictions':img})
                    gc.collect()
            except Exception as e:
                print(f"Failed to process {fragment_id}: {e}")
    except Exception as e:
        print(f"Final Exception: {e}")
    finally:
        try:
            del test_loader
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish()

        # Explicitly shut down the DataParallel model
        if isinstance(model, DataParallel):
            model = model.module  # Extract the original model
        model.cpu()  # Move the model to CPU
        del model  # Delete the model to free up GPU memory

        torch.cuda.empty_cache()  # Clean up GPU memory again
