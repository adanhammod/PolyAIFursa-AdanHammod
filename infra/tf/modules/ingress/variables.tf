variable "vpc_id" {
  description = "ID of the VPC where the ALB and its security group will be created"
  type        = string
}

variable "public_subnet_ids" {
  description = "IDs of the public subnets the ALB will be deployed into"
  type        = list(string)
}

variable "cluster_security_group_id" {
  description = "ID of the existing Kubernetes cluster security group; a rule is added to it to allow ALB traffic through to the worker nodes on the ingress-nginx NodePort"
  type        = string
}

variable "worker_asg_name" {
  description = "Name of the existing Kubernetes worker Auto Scaling Group to attach to the ALB target group"
  type        = string
}
