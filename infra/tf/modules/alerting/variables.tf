variable "topic_name" {
  description = "SNS topic name for Alertmanager alerts"
  type        = string
}

variable "alert_email" {
  description = "Email address subscribed to the Alertmanager SNS topic"
  type        = string
}
