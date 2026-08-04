output "control_plane_public_ip" {
  description = "Public IP address of the Kubernetes control plane"
  value       = aws_instance.control_plane.public_ip
}

output "vpc_id" {
  description = "ID of the Kubernetes VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets used by the Kubernetes cluster"
  value = [
    aws_subnet.public_1.id,
    aws_subnet.public_2.id
  ]
}

output "worker_asg_name" {
  description = "Name of the Kubernetes worker Auto Scaling Group"
  value       = aws_autoscaling_group.workers.name
}

output "cluster_security_group_id" {
  description = "ID of the shared Kubernetes cluster security group"
  value       = aws_security_group.cluster.id
}