module "k8s_cluster" {
  source = "./modules/k8s-cluster"

  aws_region       = var.aws_region
  ssh_allowed_cidr = var.ssh_allowed_cidr
  instance_type    = var.instance_type
  key_name         = var.key_name
  ami_id           = var.ami_id
}