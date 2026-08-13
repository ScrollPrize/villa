# Vesuvius Challenge Cloud Image

Repository for managing cloud images for [Vesuvius Challenge](https://scrollprize.org)

## Set up

1. Install Packer - instructions can be found [here](https://developer.hashicorp.com/packer/install)
    * Install Amazon plugin for `packer` ([Reference](https://developer.hashicorp.com/packer/integrations/hashicorp/amazon))
    ```bash
    packer plugins install github.com/hashicorp/amazon
    ```

2. Set up AWS CLI
    1. Obtain credentials (access keys)
    2. `pip install awscli`
    3. `aws configure`

## Create AMI

Validate config:

```bash
packer validate .
```

Create AMI:
```bash
packer build .
```

## List AMIs

```bash
aws ec2 describe-images --owners self --query 'Images[*].{ID:ImageId,Name:Name,State:State}' --output table
```

## Copy AMI to another region

Example command to move AMI from `us-east-1` (default) to `us-west-2`:

```bash
aws ec2 copy-image --source-image-id <ami-id> --source-region us-east-1 --region us-west-2 --name "vesuvius-challenge-cloud-image"
```

## Remove AMI

```bash
aws ec2 deregister-image --image-id <ami-id>
```

