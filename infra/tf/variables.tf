variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "Public IP address allowed to access the cluster"
  type        = string

  validation {
    condition     = can(cidrhost(var.ssh_allowed_cidr, 0))
    error_message = "ssh_allowed_cidr must be a valid CIDR block."
  }
}
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "key_name" {
  description = "Existing AWS EC2 key pair name"
  type        = string
}