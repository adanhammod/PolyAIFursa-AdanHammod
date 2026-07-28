variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "Public IP address allowed to access the cluster"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "key_name" {
  description = "Existing AWS EC2 key pair name"
  type        = string
}

variable "ami_id" {
  description = "Ubuntu AMI ID used by control plane and worker nodes"
  type        = string
}