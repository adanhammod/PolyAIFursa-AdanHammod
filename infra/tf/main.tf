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

module "dns" {
  source = "./modules/dns"

  zone_name    = "fursa.click"
  record_name  = "dev.adan.fursa.click"
  alb_dns_name = module.ingress.alb_dns_name
  alb_zone_id  = module.ingress.alb_zone_id
}

module "tls" {
  source = "./modules/tls"

  domain_name      = "dev.adan.fursa.click"
  zone_name        = "fursa.click"
  alb_arn          = module.ingress.alb_arn
  target_group_arn = module.ingress.target_group_arn
}