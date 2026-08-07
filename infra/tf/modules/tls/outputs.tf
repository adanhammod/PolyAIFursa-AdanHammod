output "certificate_arn" {
  description = "ARN of the validated ACM certificate"
  value       = aws_acm_certificate_validation.this.certificate_arn
}

output "https_listener_arn" {
  description = "ARN of the HTTPS ALB listener"
  value       = aws_lb_listener.https.arn
}
