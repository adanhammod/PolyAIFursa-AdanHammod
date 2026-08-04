module "k8s_cluster" {
  source = "./modules/k8s-cluster"

  aws_region       = var.aws_region
  ssh_allowed_cidr = var.ssh_allowed_cidr
  instance_type    = var.instance_type
  key_name         = var.key_name
  ami_id           = var.ami_id
}

module "ingress" {
  source = "./modules/ingress"

  vpc_id                    = module.k8s_cluster.vpc_id
  public_subnet_ids         = module.k8s_cluster.public_subnet_ids
  cluster_security_group_id = module.k8s_cluster.cluster_security_group_id
  worker_asg_name           = module.k8s_cluster.worker_asg_name
}