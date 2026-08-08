output "topic_arn" {
  description = "ARN of the SNS topic used by Alertmanager"
  value       = aws_sns_topic.alerts.arn
}

output "topic_name" {
  description = "Name of the SNS topic used by Alertmanager"
  value       = aws_sns_topic.alerts.name
}
