source "amazon-ebs" "gpu_instance" { # replace with small_instance
  ami_name      = var.ami_name
  instance_type = var.aws_instance_type
  region        = var.aws_region

  #source_ami_filter {

  #  filters = {
  #    name                = "ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"
  #    root-device-type    = "ebs"
  #    virtualization-type = "hvm"
  #  }
  #  most_recent = true
  # owners      = ["099720109477"] # Canonical
  #}
  
  #Uncomment this and comment before for GPU Instance
  source_ami_filter {
    filters = {
      name                = "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Ubuntu 20.04) 20240818"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["898082745236"] # AWS Deep Learning AMIs
  }

  ssh_username = "ubuntu"
  # TODO: Uncomment the below when we want to make the AMI publicly available
  # ami_groups   = ["all"]  # Make the AMI public

  launch_block_device_mappings {
    device_name = "/dev/sda1"
    # TODO: change this accordingly
    volume_size           = 75 # in GB, ~0.08 USD/GB/month... the PyTorch AMI requires 45GB but free tier up to 30GB?
    delete_on_termination = true
    volume_type           = "gp3"
  }
}

build {
  sources = ["source.amazon-ebs.gpu_instance"] # or small_instance

  # Run the set up script

  provisioner "shell" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install -y cloud-guest-utils",
      "echo 'df -h'",
      "df -h",
      "echo 'lsblk'",
      "lsblk"
    ]
  }

  provisioner "shell" {
    script = "scripts/install_dependencies.sh"
  }

  provisioner "shell" {
    script = "scripts/install_repositories.sh"
  }

  provisioner "shell" {
    script = "scripts/load_s3disk.sh"
  }

}