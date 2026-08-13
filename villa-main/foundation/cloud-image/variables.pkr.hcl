variable "ami_name" {
  description = "The name of the AMI"
  type        = string
  default     = "vesuvius-challenge-cloud-image-gpu"
}

variable "aws_region" {
  description = "The AWS region to use for the AMI"
  type        = string
  default     = "us-east-1" # N. Virginia
}

#variable "aws_instance_type" {
#  description = "The instance type to use for the AMI"
#  type        = string
#  default     = "t2.micro"
#}

# Uncomment this and comment the type before to use GPU
variable "aws_instance_type" {
  description = "The instance type to use for the AMI"
  type        = string
  default     = "g4dn.xlarge"
}