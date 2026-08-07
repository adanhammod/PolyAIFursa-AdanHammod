variable "zone_name" {
  description = "Name of the existing public Route53 hosted zone"
  type        = string
}

variable "record_name" {
  description = "Fully qualified DNS record name that points to the ingress ALB"
  type        = string
}

variable "alb_dns_name" {
  description = "DNS name of the ingress ALB"
  type        = string
}

variable "alb_zone_id" {
  description = "Canonical hosted zone ID of the ingress ALB"
  type        = string
}
