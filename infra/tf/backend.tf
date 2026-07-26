terraform {
  backend "s3" {
    bucket       = "adan-k8s-terraform-state-228281126655"
    key          = "kubernetes/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}