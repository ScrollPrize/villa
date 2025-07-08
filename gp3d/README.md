# GP3D: Vesuvius 2023 Grand Prize Ink Detection 3D Version

GP3D is an adaptation of the 2023 GP ind detection code to do ink detection on 3d labels for a surface volume

## How do I use it?

If you are using a fresh AWS instance such as a `g6` or `g6e` then run

```bash
cd ~
git clone https://github.com/ScrollPrize/villa.git
cd villa/gp3d
sudo bash setup_sudo.sh
bash setup_user.sh
source ~/.bashrc
```
You will need to reboot after this step

```bash
sudo reboot
```

The initial setup scripts will create a directory under `/vesuvius` that will act as the root directory for all of our work. If

```bash
cd /vesuvius
sudo chown -R ubuntu .
mkdir inkdet_outputs
mkdir inklabels
mkdir fragments
git clone https://github.com/ScrollPrize/villa.git
cd villa/gp3d
pip install -r requirements.txt
pip install --no-build-isolation transformer_engine[pytorch]
cd inkdet_outputs
wget https://dl.ash2txt.org/community-uploads/forrest/resnet3d50_epoch%3D76.ckpt
```

This will install the python requirements, setup the directory structure, and grab the latest model checkpoint

### How do I pretrain?

The GP2023 dataset is ~500GB of raw downloads. If you want to pretrain on the _entire Vesuivus dataset_, then do

```bash
cd /vesuvius
sudo apt install rclone
bash download_raw.sh
```

### How do I finetune on WebKnossos data

```bash
cd /vesuvius/villa/gp3d/
python3 wk.py --surface_volume < path to your surface volume > --ink_labels < path to your inklabels >
```

e.g.

```bash
python3 wk.py --surface_volume http://dl.ash2txt.org:8080/data/annotations/zarr/YRkpZecXiB57JOGm/surface_volume/1/ --ink_labels http://dl.ash2txt.org:8080/data/annotations/zarr/YRkpZecXiB57JOGm/ink_labels/1/
```

This will download the surface volume and ink labels. If you see `All chunks successfully downloaded!` after both progress bars, then the downloads completely successfully. 

Next we need to download the most recent checkpoint. Navigate to https://dl.ash2txt.org/community-uploads/forrest/ and find the `resnet3d50_epoch=*.ckpt` file with the highest checkpoint number then `wget` that, e.g.

```bash
cd /vesuvius/inkdet_outputs/
wget https://dl.ash2txt.org/community-uploads/forrest/resnet3d50_epoch%3D76.ckpt
```

Finally, run the training script

```bash
cd /vesuvius/villa/gp3d
python3 train.py
```
If the training is working properly, then you will see, among other things, a message stating that we are resuming from a checkpoint

```
Resuming from checkpoint: /vesuvius//inkdet_outputs/resnet3d50_epoch=76.ckpt
```

The amount of fragments and which ones they are will also be displayed

```
Total fragments: 1
Train fragments: 1
Valid fragments: 0
Loaded 3D ink mask from zarr for YRkpZecXiB57JOGm, shape: (64, 10944, 29888)
```

After a couple minutes of setup, training will begin and output a progress bar like

```
Epoch 77:  10%|██████▌    | 76/727 [03:58<33:59,  0.32it/s, v_num=1, train/loss_step=0.580, train/lr_step=0.0008, train/epoch_step=77.00, train/step_step=35464.0]
```

You are now fine tuning on your data from webknossos!

To stop training, you can `ctrl+c` to kill the python training process. 

### How do I run inference

Ensure that your fragment zarr exists under `/vesuvius/fragments`, e.g. `/vesuvius/fragments/YRkpZecXiB57JOGm.zarr` and run

```bash
cd /vesuvius/villa/gp3d
python3 --checkpoint_path < path to your checkpoint > --fragment_id < your fragment id>
```

e.g.

```bash
python3 infer.py --checkpoint_path /vesuvius/inkdet_outputs/resnet3d50_epoch=76.ckpt --fragment_id YRkpZecXiB57JOGm
```