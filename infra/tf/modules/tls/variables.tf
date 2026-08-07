variable "domain_name" {
  description = "Domain name protected by the ACM certificate"
  type        = string
}

variable "zone_name" {
  description = "Name of the existing public Route53 hosted zone"
  type        = string
}

variable "alb_arn" {
  description = "ARN of the existing ingress ALB"
  type        = string
}

variable "target_group_arn" {
  description = "ARN of the existing ingress target group"
  type        = string
}
