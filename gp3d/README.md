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

This will create a directory under `/vesuvius` that will act as the root directory for all of our work. If

```bash
cd /vesuvius
sudo chown -R ubuntu .
mkdir inkdet_outputs
mkdir inklabels
mkdir fragments
git clone https://github.com/ScrollPrize/villa.git
cd villa/gp3d
pip install -r requirements.txt
cd inkdet_outputs
wget https://dl.ash2txt.org/community-uploads/forrest/resnet3d50_epoch%3D76.ckpt
```

This will install the python requirements, setup the directory structure, and grab the latest model checkpoint

### How do I pretrain?

If you want to pretrain on the _entire Vesuivus dataset_, then do

```bash
cd /vesuvius
sudo apt install rclone
bash download_raw.sh

```