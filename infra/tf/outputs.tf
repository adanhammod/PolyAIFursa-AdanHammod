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