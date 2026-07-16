import os
from typing import Any

import boto3
from fastmcp import FastMCP
import requests

mcp = FastMCP("observability")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DEV_S3_LOGS_BUCKET = os.getenv(
    "DEV_S3_LOGS_BUCKET",
    "adan-polyai-logs-dev",
)
PROD_S3_LOGS_BUCKET = os.getenv(
    "PROD_S3_LOGS_BUCKET",
    "adan-polyai-logs-prod",
)

DEV_PROMETHEUS_URL = os.getenv(
    "DEV_PROMETHEUS_URL",
    "http://adan-dev.fursa.click:9090",
)

PROD_PROMETHEUS_URL = os.getenv(
    "PROD_PROMETHEUS_URL",
    "http://prod.adan.fursa.click:9090",
)

s3_client = boto3.client("s3", region_name=AWS_REGION)


def _get_logs_bucket(environment: str) -> str:
    normalized_environment = environment.strip().lower()

    if normalized_environment == "dev":
        return DEV_S3_LOGS_BUCKET

    if normalized_environment == "prod":
        return PROD_S3_LOGS_BUCKET

    raise ValueError("Invalid environment. Use 'dev' or 'prod'.")


def _get_prometheus_url(environment: str) -> str:
    normalized_environment = environment.strip().lower()

    if normalized_environment == "dev":
        return DEV_PROMETHEUS_URL

    if normalized_environment == "prod":
        return PROD_PROMETHEUS_URL

    raise ValueError("Invalid environment. Use 'dev' or 'prod'.")


@mcp.tool
def health() -> str:
    """Check whether the Observability MCP server is running."""
    return "Observability MCP is running"


@mcp.tool
def list_log_files(
    environment: str = "dev",
    max_results: int = 20,
) -> dict[str, Any]:
    """
    List recent container-log objects stored in the S3 bucket
    for the selected environment.
    """
    if max_results < 1 or max_results > 100:
        raise ValueError("max_results must be between 1 and 100")

    bucket = _get_logs_bucket(environment)

    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix="logs/",
        MaxKeys=max_results,
    )

    objects = response.get("Contents", [])

    files = [
        {
            "key": item["Key"],
            "size_bytes": item["Size"],
            "last_modified": item["LastModified"].isoformat(),
        }
        for item in objects
    ]

    return {
        "environment": environment.lower(),
        "bucket": bucket,
        "count": len(files),
        "files": files,
    }


@mcp.tool
def query_prometheus(
    query: str,
    environment: str = "dev",
) -> dict[str, Any]:
    """
    Execute a PromQL instant query against Prometheus for the
    selected environment.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    prometheus_url = _get_prometheus_url(environment)

    response = requests.get(
        f"{prometheus_url}/api/v1/query",
        params={"query": query},
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")

    result_data = payload.get("data", {})

    return {
        "environment": environment.lower(),
        "prometheus_url": prometheus_url,
        "query": query,
        "result_type": result_data.get("resultType"),
        "result": result_data.get("result"),
    }


if __name__ == "__main__":
    mcp.run()
