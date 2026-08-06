output "record_fqdn" {
  description = "Fully qualified Route53 record created for the ingress ALB"
  value       = aws_route53_record.this.fqdn
}

output "hosted_zone_id" {
  description = "ID of the existing Route53 hosted zone used by the record"
  value       = data.aws_route53_zone.this.zone_id
}
