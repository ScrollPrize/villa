# Install s3fs and set up mount points
sudo apt install -y s3fs

# Create directories where the S3 buckets will be mounted
sudo mkdir -p /mnt/scrolls
sudo mkdir -p /mnt/annotated-instances

# Set up AWS credentials for s3fs
# echo ACCESS_KEY_ID:SECRET_ACCESS_KEY > /home/ubuntu/.passwd-s3fs
# sudo chmod 600 /home/ubuntu/.passwd-s3fs

# Load fuse module
echo "fuse" | sudo tee -a /etc/modules

# Test the mount manually to ensure everything is working, the first two lines have password
#sudo s3fs herculaneum-scrolls /mnt/scrolls -o ro,use_cache=/tmp,allow_other,use_path_request_style,url=https://s3.amazonaws.com,passwd_file=/home/ubuntu/.passwd-s3fs,uid=1000,gid=1000,umask=0022,mp_umask=0022
#sudo s3fs herculaneum-annotated-instances /mnt/annotated-instances -o ro,use_cache=/tmp,allow_other,use_path_request_style,url=https://s3.amazonaws.com,passwd_file=/home/ubuntu/.passwd-s3fs,uid=1000,gid=1000,umask=0022,mp_umask=0022
sudo s3fs herculaneum-scrolls /mnt/scrolls -o public_bucket=1,ro,use_cache=/tmp,allow_other,use_path_request_style,url=https://s3.amazonaws.com,uid=1000,gid=1000,umask=0022,mp_umask=0022
sudo s3fs herculaneum-annotated-instances /mnt/annotated-instances -o public_bucket=1,ro,use_cache=/tmp,allow_other,use_path_request_style,url=https://s3.amazonaws.com,uid=1000,gid=1000,umask=0022,mp_umask=0022

# Create a systemd service file to mount the S3 buckets at startup
sudo bash -c 'cat <<EOF > /etc/systemd/system/mount-s3.service
[Unit]
Description=Mount S3 Buckets at startup
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/s3fs herculaneum-scrolls /mnt/scrolls -o public_bucket=1,ro,use_cache=/tmp,allow_other,use_path_request_style,url=https://s3.amazonaws.com,uid=1000,gid=1000,umask=0022,mp_umask=0022
ExecStart=/usr/bin/s3fs herculaneum-annotated-instances /mnt/annotated-instances -o public_bucket=1,ro,use_cache=/tmp,allow_other,use_path_request_style,url=https://s3.amazonaws.com,uid=1000,gid=1000,umask=0022,mp_umask=0022
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF'

# Enable the service so it runs at every boot
sudo systemctl enable mount-s3.service

# Proceed with the rest of your script
# Ensure Conda is available
MINICONDA_PREFIX=/usr/local/miniconda
eval "$($MINICONDA_PREFIX/bin/conda shell.bash hook)"
source ~/.bashrc
export PATH=$MINICONDA_PREFIX/bin:$PATH

# Vesuvius
echo "Installing vesuvius scaffold..."
conda create -n vesuvius python=3.12 -y || log_and_exit "Failed to create Vesuvius environment"
conda activate vesuvius
conda install jupyter matplotlib
pip install vesuvius
vesuvius.accept_terms --yes

echo "Installation complete"
