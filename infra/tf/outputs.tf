output "control_plane_public_ip" {
  description = "Public IP address of the Kubernetes control plane"
  value       = module.k8s_cluster.control_plane_public_ip
}

output "vpc_id" {
  description = "ID of the Kubernetes VPC"
  value       = module.k8s_cluster.vpc_id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets used by the Kubernetes cluster"
  value       = module.k8s_cluster.public_subnet_ids
}

output "worker_asg_name" {
  description = "Name of the Kubernetes worker Auto Scaling Group"
  value       = module.k8s_cluster.worker_asg_name
}

output "cluster_security_group_id" {
  description = "ID of the shared Kubernetes cluster security group"
  value       = module.k8s_cluster.cluster_security_group_id
}

output "alb_dns_name" {
  description = "DNS name of the ingress ALB"
  value       = module.ingress.alb_dns_name
}

output "alb_zone_id" {
  description = "Canonical hosted zone ID of the ingress ALB"
  value       = module.ingress.alb_zone_id
}

output "alerting_sns_topic_arn" {
  description = "SNS topic ARN used by Alertmanager"
  value       = module.alerting.topic_arn
}

output "alerting_sns_topic_name" {
  description = "SNS topic name used by Alertmanager"
  value       = module.alerting.topic_name
}