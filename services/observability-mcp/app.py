from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from fastmcp import FastMCP
from requests import Response
from requests.exceptions import RequestException


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
).rstrip("/")
PROD_PROMETHEUS_URL = os.getenv(
    "PROD_PROMETHEUS_URL",
    "http://prod.adan.fursa.click:9090",
).rstrip("/")

S3_LOG_PREFIX = "logs/"
REQUEST_TIMEOUT_SECONDS = 15
DEFAULT_MAX_LINES = 200
HARD_MAX_LINES = 1000
MAX_LOG_LINE_LENGTH = 4000

ERROR_KEYWORDS = (
    "error",
    "exception",
    "traceback",
    "failed",
    "failure",
    "fatal",
    "500",
)

SERVICE_FIELDS = (
    "container_name",
    "service",
    "container",
    "container_name_label",
    "com.docker.compose.service",
)

s3_client = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Validation and environment helpers
# ---------------------------------------------------------------------------


def _normalize_environment(environment: str) -> str:
    """Validate and normalize an environment name."""
    if not isinstance(environment, str):
        raise ValueError("environment must be a string")

    normalized = environment.strip().lower()

    if normalized not in {"dev", "prod"}:
        raise ValueError("Invalid environment. Use 'dev' or 'prod'.")

    return normalized


def _get_logs_bucket(environment: str) -> str:
    """Return the configured S3 logs bucket for an environment."""
    normalized = _normalize_environment(environment)

    if normalized == "dev":
        return DEV_S3_LOGS_BUCKET

    return PROD_S3_LOGS_BUCKET


def _get_prometheus_url(environment: str) -> str:
    """Return the configured Prometheus URL for an environment."""
    normalized = _normalize_environment(environment)

    if normalized == "dev":
        return DEV_PROMETHEUS_URL

    return PROD_PROMETHEUS_URL


def _validate_int_range(
    name: str,
    value: int,
    minimum: int,
    maximum: int,
) -> None:
    """Validate that an integer is inside an inclusive range."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")

    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _require_non_empty(name: str, value: str) -> str:
    """Validate and return a stripped, non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")

    return value.strip()


def _safe_float(value: Any) -> float | None:
    """Convert a Prometheus value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate_line(line: str) -> str:
    """Prevent a single log entry from creating an oversized tool result."""
    if len(line) <= MAX_LOG_LINE_LENGTH:
        return line

    return f"{line[:MAX_LOG_LINE_LENGTH]}…"


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _list_log_objects(
    bucket: str,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    maximum: int | None = None,
) -> list[dict[str, Any]]:
    """
    List S3 log objects, optionally filtered by modification time.

    Results are sorted newest first.
    """
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        objects: list[dict[str, Any]] = []

        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=S3_LOG_PREFIX,
        ):
            for item in page.get("Contents", []):
                modified = item.get("LastModified")

                if modified is None:
                    continue

                if start_time is not None and modified < start_time:
                    continue

                if end_time is not None and modified > end_time:
                    continue

                objects.append(item)

        objects.sort(
            key=lambda item: item["LastModified"],
            reverse=True,
        )

        if maximum is not None:
            return objects[:maximum]

        return objects

    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(
            f"Failed to list log objects in S3 bucket '{bucket}'."
        ) from exc


def _list_recent_log_objects(
    bucket: str,
    minutes: int,
    max_files: int,
) -> list[dict[str, Any]]:
    """List recent log objects within a rolling time window."""
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(minutes=minutes)

    return _list_log_objects(
        bucket,
        start_time=start_time,
        end_time=now,
        maximum=max_files,
    )


def _download_and_decompress_log(
    bucket: str,
    key: str,
) -> str:
    """Download and decompress one gzip S3 log object."""
    normalized_key = _require_non_empty("key", key)

    if not normalized_key.startswith(S3_LOG_PREFIX):
        raise ValueError(f"key must start with '{S3_LOG_PREFIX}'")

    try:
        response = s3_client.get_object(
            Bucket=bucket,
            Key=normalized_key,
        )
        compressed_data = response["Body"].read()

    except (ClientError, BotoCoreError, KeyError) as exc:
        raise RuntimeError(
            f"Failed to download S3 log object '{normalized_key}'."
        ) from exc

    try:
        decompressed = gzip.decompress(compressed_data)
    except (OSError, EOFError) as exc:
        raise RuntimeError(
            f"S3 object '{normalized_key}' is not valid gzip data."
        ) from exc

    return decompressed.decode("utf-8", errors="replace")


def _parse_log_line(line: str) -> dict[str, Any]:
    """Parse a Docker JSON log line when possible."""
    stripped = line.strip()

    if not stripped:
        return {"raw": ""}

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {"raw": stripped}

    if not isinstance(payload, dict):
        return {
            "raw": stripped,
            "value": payload,
        }

    payload.setdefault("raw", stripped)
    return payload


def _parse_attrs(value: Any) -> dict[str, Any]:
    """Normalize Docker attrs whether represented as an object or string."""
    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value.strip():
        return {}

    stripped = value.strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    attrs: dict[str, str] = {}

    for part in stripped.split(","):
        if "=" not in part:
            continue

        key, raw_value = part.split("=", 1)
        attrs[key.strip()] = raw_value.strip()

    return attrs


def _normalize_service_name(value: Any) -> str | None:
    """Normalize a discovered service label."""
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def _extract_service_name(
    parsed: dict[str, Any],
    raw_line: str = "",
) -> str | None:
    """Extract a service name from common Docker log metadata fields."""
    for field in SERVICE_FIELDS:
        service = _normalize_service_name(parsed.get(field))
        if service:
            return service

    attrs = _parse_attrs(parsed.get("attrs"))

    for field in SERVICE_FIELDS:
        service = _normalize_service_name(attrs.get(field))
        if service:
            return service

    labels = parsed.get("labels")

    if isinstance(labels, dict):
        for field in SERVICE_FIELDS:
            service = _normalize_service_name(labels.get(field))
            if service:
                return service

    compose_service = parsed.get("com.docker.compose.service")

    if compose_service:
        return _normalize_service_name(compose_service)

    # Plain-text fallback for known field representations.
    match = re.search(
        r"(?i)(?:container_name|service|container)"
        r'["\s:=]+([a-z0-9_.-]+)',
        raw_line,
    )

    if match:
        return match.group(1)

    return None


def _extract_log_message(
    parsed: dict[str, Any],
    raw_line: str,
) -> str:
    """Extract the application log message from a Docker JSON record."""
    for field in ("log", "message", "msg"):
        value = parsed.get(field)

        if isinstance(value, str):
            return value.rstrip("\r\n")

    return raw_line.rstrip("\r\n")


def _service_matches(
    requested_service: str,
    discovered_service: str | None,
    parsed: dict[str, Any],
    raw_line: str,
) -> bool:
    """Return whether a log record belongs to the requested service."""
    requested = requested_service.strip().lower()

    if discovered_service:
        discovered = discovered_service.lower()

        if requested == discovered:
            return True

        if requested in discovered or discovered in requested:
            return True

    searchable = json.dumps(
        parsed,
        ensure_ascii=False,
        default=str,
    ).lower()

    return requested in searchable or requested in raw_line.lower()


def _search_log_entries(
    *,
    environment: str,
    keywords: tuple[str, ...],
    service: str | None,
    start_time: datetime,
    end_time: datetime,
    max_files: int,
    max_matches: int,
) -> dict[str, Any]:
    """Search gzip log objects for one or more case-insensitive keywords."""
    normalized_environment = _normalize_environment(environment)
    bucket = _get_logs_bucket(normalized_environment)

    lowered_keywords = tuple(
        keyword.strip().lower() for keyword in keywords if keyword.strip()
    )

    if not lowered_keywords:
        raise ValueError("At least one search keyword is required")

    normalized_service = (
        _require_non_empty("service", service) if service is not None else None
    )

    objects = _list_log_objects(
        bucket,
        start_time=start_time,
        end_time=end_time,
        maximum=max_files,
    )

    matches: list[dict[str, Any]] = []
    files_scanned = 0

    for item in objects:
        key = item["Key"]
        files_scanned += 1

        try:
            text = _download_and_decompress_log(bucket, key)
        except RuntimeError:
            # One damaged/unavailable file should not prevent all results.
            continue

        for raw_line in text.splitlines():
            parsed = _parse_log_line(raw_line)
            message = _extract_log_message(parsed, raw_line)
            service_name = _extract_service_name(parsed, raw_line)

            if normalized_service and not _service_matches(
                normalized_service,
                service_name,
                parsed,
                raw_line,
            ):
                continue

            searchable = f"{message}\n{raw_line}".lower()

            matched_keywords = [
                keyword for keyword in lowered_keywords if keyword in searchable
            ]

            if not matched_keywords:
                continue

            matches.append(
                {
                    "key": key,
                    "last_modified": item["LastModified"].isoformat(),
                    "service": service_name,
                    "matched_keywords": matched_keywords,
                    "line": _truncate_line(message or raw_line),
                }
            )

            if len(matches) >= max_matches:
                break

        if len(matches) >= max_matches:
            break

    return {
        "environment": normalized_environment,
        "bucket": bucket,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "service": normalized_service,
        "files_scanned": files_scanned,
        "match_count": len(matches),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------


def _prometheus_request(
    environment: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a request to a Prometheus HTTP API endpoint."""
    normalized_environment = _normalize_environment(environment)
    prometheus_url = _get_prometheus_url(normalized_environment)
    url = f"{prometheus_url}{path}"

    try:
        response: Response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

    except RequestException as exc:
        raise RuntimeError(
            f"Failed to reach Prometheus for environment '{normalized_environment}'."
        ) from exc
    except ValueError as exc:
        raise RuntimeError("Prometheus returned an invalid JSON response.") from exc

    if payload.get("status") != "success":
        error_type = payload.get("errorType", "unknown")
        error_message = payload.get(
            "error",
            "Prometheus request failed",
        )
        raise RuntimeError(f"Prometheus error ({error_type}): {error_message}")

    return payload


def _prometheus_instant_query(
    query: str,
    environment: str,
    *,
    evaluation_time: datetime | None = None,
) -> dict[str, Any]:
    """Execute a Prometheus instant query."""
    normalized_query = _require_non_empty("query", query)

    params: dict[str, Any] = {
        "query": normalized_query,
    }

    if evaluation_time is not None:
        params["time"] = evaluation_time.timestamp()

    payload = _prometheus_request(
        environment,
        "/api/v1/query",
        params=params,
    )

    data = payload.get("data", {})

    return {
        "result_type": data.get("resultType"),
        "result": data.get("result", []),
    }


def _prometheus_range_query(
    query: str,
    environment: str,
    *,
    start_time: datetime,
    end_time: datetime,
    step_seconds: int,
) -> dict[str, Any]:
    """Execute a Prometheus range query."""
    normalized_query = _require_non_empty("query", query)

    payload = _prometheus_request(
        environment,
        "/api/v1/query_range",
        params={
            "query": normalized_query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": step_seconds,
        },
    )

    data = payload.get("data", {})

    return {
        "result_type": data.get("resultType"),
        "result": data.get("result", []),
    }


def _escape_promql_label_value(value: str) -> str:
    """Escape a value placed inside a PromQL quoted label matcher."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _service_health_data(
    service: str,
    environment: str,
) -> dict[str, Any]:
    """Return Prometheus health data for one service."""
    normalized_service = _require_non_empty(
        "service",
        service,
    )
    normalized_environment = _normalize_environment(environment)
    escaped_service = _escape_promql_label_value(normalized_service)

    exact_query = f'up{{job="{escaped_service}"}}'
    query_result = _prometheus_instant_query(
        exact_query,
        normalized_environment,
    )
    matching_targets = query_result["result"]

    query_used = exact_query

    if not matching_targets:
        regex_service = re.escape(normalized_service).replace(
            "\\-",
            "-",
        )
        fallback_query = f'up{{instance=~".*{regex_service}.*"}}'
        query_result = _prometheus_instant_query(
            fallback_query,
            normalized_environment,
        )
        matching_targets = query_result["result"]
        query_used = fallback_query

    values: list[float] = []
    targets: list[dict[str, Any]] = []

    for item in matching_targets:
        raw_value = item.get("value", [None, None])
        numeric_value = (
            _safe_float(raw_value[1])
            if isinstance(raw_value, list) and len(raw_value) >= 2
            else None
        )

        if numeric_value is not None:
            values.append(numeric_value)

        targets.append(
            {
                "metric": item.get("metric", {}),
                "value": numeric_value,
                "timestamp": (
                    raw_value[0] if isinstance(raw_value, list) and raw_value else None
                ),
            }
        )

    if not values:
        status = "unknown"
    elif any(value >= 1 for value in values):
        status = "up"
    else:
        status = "down"

    return {
        "service": normalized_service,
        "environment": normalized_environment,
        "status": status,
        "query": query_used,
        "matching_targets": targets,
        "raw_values": values,
    }


def _cpu_usage_data(
    service: str,
    environment: str,
    minutes: int,
) -> dict[str, Any]:
    """Return CPU usage metrics for one service."""
    normalized_service = _require_non_empty(
        "service",
        service,
    )
    normalized_environment = _normalize_environment(environment)
    escaped_service = _escape_promql_label_value(normalized_service)

    is_node_exporter = normalized_service.lower() in {
        "node-exporter",
        "node_exporter",
        "node",
    }

    if is_node_exporter:
        query = (
            "100 - ("
            "avg by(instance) ("
            "rate(node_cpu_seconds_total"
            f'{{job="node-exporter",mode="idle"}}'
            f"[{minutes}m])"
            ") * 100"
            ")"
        )
        unit = "percent"
    else:
        query = (
            f'rate(process_cpu_seconds_total{{job="{escaped_service}"}}[{minutes}m])'
        )
        unit = "cpu_cores"

    query_result = _prometheus_instant_query(
        query,
        normalized_environment,
    )

    values: list[dict[str, Any]] = []

    for item in query_result["result"]:
        raw_value = item.get("value", [None, None])
        numeric_value = (
            _safe_float(raw_value[1])
            if isinstance(raw_value, list) and len(raw_value) >= 2
            else None
        )

        values.append(
            {
                "metric": item.get("metric", {}),
                "value": numeric_value,
            }
        )

    return {
        "service": normalized_service,
        "environment": normalized_environment,
        "minutes": minutes,
        "query": query,
        "unit": unit,
        "values": values,
        "result_count": len(values),
    }


def _memory_usage_data(
    service: str,
    environment: str,
) -> dict[str, Any]:
    """Return memory usage metrics for one service."""
    normalized_service = _require_non_empty(
        "service",
        service,
    )
    normalized_environment = _normalize_environment(environment)
    escaped_service = _escape_promql_label_value(normalized_service)

    is_node_exporter = normalized_service.lower() in {
        "node-exporter",
        "node_exporter",
        "node",
    }

    if is_node_exporter:
        query = (
            "node_memory_MemTotal_bytes"
            '{job="node-exporter"}'
            " - "
            "node_memory_MemAvailable_bytes"
            '{job="node-exporter"}'
        )
    else:
        query = f'process_resident_memory_bytes{{job="{escaped_service}"}}'

    query_result = _prometheus_instant_query(
        query,
        normalized_environment,
    )

    values: list[dict[str, Any]] = []

    for item in query_result["result"]:
        raw_value = item.get("value", [None, None])
        bytes_value = (
            _safe_float(raw_value[1])
            if isinstance(raw_value, list) and len(raw_value) >= 2
            else None
        )

        values.append(
            {
                "metric": item.get("metric", {}),
                "bytes": bytes_value,
                "mib": (
                    bytes_value / (1024 * 1024) if bytes_value is not None else None
                ),
            }
        )

    return {
        "service": normalized_service,
        "environment": normalized_environment,
        "query": query,
        "unit": "bytes_and_mib",
        "values": values,
        "result_count": len(values),
    }


def _recent_errors_data(
    service: str,
    environment: str,
    minutes: int,
    max_matches: int,
) -> dict[str, Any]:
    """Search recent service logs for common error indicators."""
    now = datetime.now(timezone.utc)

    result = _search_log_entries(
        environment=environment,
        keywords=ERROR_KEYWORDS,
        service=service,
        start_time=now - timedelta(minutes=minutes),
        end_time=now,
        max_files=100,
        max_matches=max_matches,
    )

    return {
        **result,
        "minutes": minutes,
        "error_indicators": list(ERROR_KEYWORDS),
    }


def _safe_section(
    callback: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Execute one diagnostic section without failing the full report."""
    try:
        return callback()
    except Exception as exc:
        return {
            "error": str(exc),
        }


def _investigate_service_data(
    service: str,
    environment: str,
    minutes: int,
) -> dict[str, Any]:
    """Combine health, CPU, memory and recent log errors."""
    normalized_service = _require_non_empty(
        "service",
        service,
    )
    normalized_environment = _normalize_environment(environment)

    return {
        "service": normalized_service,
        "environment": normalized_environment,
        "minutes": minutes,
        "health": _safe_section(
            lambda: _service_health_data(
                normalized_service,
                normalized_environment,
            )
        ),
        "cpu": _safe_section(
            lambda: _cpu_usage_data(
                normalized_service,
                normalized_environment,
                min(minutes, 60),
            )
        ),
        "memory": _safe_section(
            lambda: _memory_usage_data(
                normalized_service,
                normalized_environment,
            )
        ),
        "recent_errors": _safe_section(
            lambda: _recent_errors_data(
                normalized_service,
                normalized_environment,
                minutes,
                100,
            )
        ),
    }


def _maximum_numeric_value(
    section: dict[str, Any],
    field: str,
) -> float | None:
    """Get the highest numeric field from a metric result section."""
    values = section.get("values")

    if not isinstance(values, list):
        return None

    numeric_values: list[float] = []

    for item in values:
        if not isinstance(item, dict):
            continue

        value = _safe_float(item.get(field))

        if value is not None:
            numeric_values.append(value)

    return max(numeric_values) if numeric_values else None


def _parse_iso_timestamp(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""
    normalized = _require_non_empty("timestamp", timestamp)

    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO-8601 value") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Basic MCP tools
# ---------------------------------------------------------------------------


@mcp.tool
def health() -> str:
    """Check whether the Observability MCP server is running."""
    return "Observability MCP is running"


@mcp.tool
def list_log_files(
    environment: str = "dev",
    max_results: int = 20,
) -> dict[str, Any]:
    """List the most recent container-log objects in S3."""
    normalized_environment = _normalize_environment(environment)
    _validate_int_range(
        "max_results",
        max_results,
        1,
        100,
    )

    bucket = _get_logs_bucket(normalized_environment)
    objects = _list_log_objects(
        bucket,
        maximum=max_results,
    )

    files = [
        {
            "key": item["Key"],
            "size_bytes": item["Size"],
            "last_modified": item["LastModified"].isoformat(),
        }
        for item in objects
    ]

    return {
        "environment": normalized_environment,
        "bucket": bucket,
        "count": len(files),
        "files": files,
    }


@mcp.tool
def query_prometheus(
    query: str,
    environment: str = "dev",
) -> dict[str, Any]:
    """Execute a PromQL instant query."""
    normalized_environment = _normalize_environment(environment)
    normalized_query = _require_non_empty("query", query)

    query_result = _prometheus_instant_query(
        normalized_query,
        normalized_environment,
    )

    return {
        "environment": normalized_environment,
        "prometheus_url": _get_prometheus_url(normalized_environment),
        "query": normalized_query,
        **query_result,
    }


# ---------------------------------------------------------------------------
# S3 log MCP tools
# ---------------------------------------------------------------------------


@mcp.tool
def read_log_file(
    key: str,
    environment: str = "dev",
    max_lines: int = DEFAULT_MAX_LINES,
) -> dict[str, Any]:
    """Read the last lines of one gzip log object from S3."""
    normalized_environment = _normalize_environment(environment)
    normalized_key = _require_non_empty("key", key)

    _validate_int_range(
        "max_lines",
        max_lines,
        1,
        HARD_MAX_LINES,
    )

    bucket = _get_logs_bucket(normalized_environment)
    text = _download_and_decompress_log(
        bucket,
        normalized_key,
    )
    all_lines = text.splitlines()
    selected_lines = all_lines[-max_lines:]

    return {
        "environment": normalized_environment,
        "bucket": bucket,
        "key": normalized_key,
        "total_lines": len(all_lines),
        "returned_lines": len(selected_lines),
        "lines": [_truncate_line(line) for line in selected_lines],
    }


@mcp.tool
def search_logs(
    keyword: str,
    environment: str = "dev",
    service: str | None = None,
    minutes: int = 5,
    max_files: int = 50,
    max_matches: int = 200,
) -> dict[str, Any]:
    """Search recent S3 container logs for a keyword."""
    normalized_keyword = _require_non_empty(
        "keyword",
        keyword,
    )

    _validate_int_range("minutes", minutes, 1, 1440)
    _validate_int_range("max_files", max_files, 1, 200)
    _validate_int_range(
        "max_matches",
        max_matches,
        1,
        HARD_MAX_LINES,
    )

    now = datetime.now(timezone.utc)

    result = _search_log_entries(
        environment=environment,
        keywords=(normalized_keyword,),
        service=service,
        start_time=now - timedelta(minutes=minutes),
        end_time=now,
        max_files=max_files,
        max_matches=max_matches,
    )

    return {
        **result,
        "keyword": normalized_keyword,
        "minutes": minutes,
    }


@mcp.tool
def list_services_from_logs(
    environment: str = "dev",
    minutes: int = 60,
    max_files: int = 100,
) -> dict[str, Any]:
    """Discover service names contained in recent Docker log records."""
    normalized_environment = _normalize_environment(environment)

    _validate_int_range("minutes", minutes, 1, 1440)
    _validate_int_range("max_files", max_files, 1, 200)

    bucket = _get_logs_bucket(normalized_environment)
    objects = _list_recent_log_objects(
        bucket,
        minutes,
        max_files,
    )

    counts: Counter[str] = Counter()
    files_scanned = 0
    entries_scanned = 0

    for item in objects:
        files_scanned += 1

        try:
            text = _download_and_decompress_log(
                bucket,
                item["Key"],
            )
        except RuntimeError:
            continue

        for raw_line in text.splitlines():
            entries_scanned += 1
            parsed = _parse_log_line(raw_line)
            service_name = _extract_service_name(
                parsed,
                raw_line,
            )

            if service_name:
                counts[service_name] += 1

    services = [
        {
            "service": service,
            "entries": count,
        }
        for service, count in counts.most_common()
    ]

    return {
        "environment": normalized_environment,
        "bucket": bucket,
        "minutes": minutes,
        "files_scanned": files_scanned,
        "entries_scanned": entries_scanned,
        "service_count": len(services),
        "services": services,
    }


@mcp.tool
def show_recent_errors(
    service: str,
    environment: str = "dev",
    minutes: int = 15,
    max_matches: int = 100,
) -> dict[str, Any]:
    """Show recent error-like log entries for one service."""
    _validate_int_range("minutes", minutes, 1, 1440)
    _validate_int_range(
        "max_matches",
        max_matches,
        1,
        HARD_MAX_LINES,
    )

    return _recent_errors_data(
        service,
        environment,
        minutes,
        max_matches,
    )


# ---------------------------------------------------------------------------
# Prometheus MCP tools
# ---------------------------------------------------------------------------


@mcp.tool
def get_target_status(
    environment: str = "dev",
) -> dict[str, Any]:
    """Return the status of active Prometheus scrape targets."""
    normalized_environment = _normalize_environment(environment)

    payload = _prometheus_request(
        normalized_environment,
        "/api/v1/targets",
    )

    active_targets = payload.get("data", {}).get("activeTargets", [])

    targets: list[dict[str, Any]] = []
    healthy = 0

    for target in active_targets:
        health_status = target.get("health", "unknown")

        if health_status == "up":
            healthy += 1

        discovered_labels = target.get(
            "discoveredLabels",
            {},
        )
        labels = target.get("labels", {})

        targets.append(
            {
                "job": labels.get("job") or discovered_labels.get("job"),
                "instance": labels.get("instance") or discovered_labels.get("instance"),
                "health": health_status,
                "last_scrape": target.get("lastScrape"),
                "last_error": target.get("lastError"),
                "scrape_url": target.get("scrapeUrl"),
            }
        )

    total = len(targets)

    return {
        "environment": normalized_environment,
        "prometheus_url": _get_prometheus_url(normalized_environment),
        "total": total,
        "healthy": healthy,
        "unhealthy": total - healthy,
        "targets": targets,
    }


@mcp.tool
def get_service_health(
    service: str,
    environment: str = "dev",
) -> dict[str, Any]:
    """Return whether a service is up, down or unknown."""
    return _service_health_data(service, environment)


@mcp.tool
def get_cpu_usage(
    service: str,
    environment: str = "dev",
    minutes: int = 5,
) -> dict[str, Any]:
    """Return recent CPU usage for a service."""
    _validate_int_range("minutes", minutes, 1, 60)

    return _cpu_usage_data(
        service,
        environment,
        minutes,
    )


@mcp.tool
def get_memory_usage(
    service: str,
    environment: str = "dev",
) -> dict[str, Any]:
    """Return current resident memory usage for a service."""
    return _memory_usage_data(service, environment)


@mcp.tool
def get_request_rate(
    service: str,
    environment: str = "dev",
    minutes: int = 5,
) -> dict[str, Any]:
    """Return request rate using the first supported request counter."""
    normalized_service = _require_non_empty(
        "service",
        service,
    )
    normalized_environment = _normalize_environment(environment)

    _validate_int_range("minutes", minutes, 1, 60)

    escaped_service = _escape_promql_label_value(normalized_service)

    candidate_metrics = (
        "http_requests_total",
        "http_server_requests_total",
        "requests_total",
    )

    attempted_queries: list[str] = []

    for metric_name in candidate_metrics:
        query = f'rate({metric_name}{{job="{escaped_service}"}}[{minutes}m])'
        attempted_queries.append(query)

        result = _prometheus_instant_query(
            query,
            normalized_environment,
        )

        if result["result"]:
            return {
                "service": normalized_service,
                "environment": normalized_environment,
                "minutes": minutes,
                "metric": metric_name,
                "query": query,
                "unit": "requests_per_second",
                "result": result["result"],
            }

    return {
        "service": normalized_service,
        "environment": normalized_environment,
        "minutes": minutes,
        "metric": None,
        "unit": "requests_per_second",
        "result": [],
        "message": ("No supported request counter returned data."),
        "attempted_queries": attempted_queries,
    }


# ---------------------------------------------------------------------------
# Combined investigation MCP tools
# ---------------------------------------------------------------------------


@mcp.tool
def investigate_service(
    service: str,
    environment: str = "dev",
    minutes: int = 15,
) -> dict[str, Any]:
    """Combine service health, CPU, memory and recent errors."""
    _validate_int_range("minutes", minutes, 1, 60)

    return _investigate_service_data(
        service,
        environment,
        minutes,
    )


@mcp.tool
def diagnose_service(
    service: str,
    environment: str = "dev",
    minutes: int = 15,
) -> dict[str, Any]:
    """Produce a deterministic diagnosis from metrics and logs."""
    _validate_int_range("minutes", minutes, 1, 60)

    investigation = _investigate_service_data(
        service,
        environment,
        minutes,
    )

    health_section = investigation.get("health", {})
    cpu_section = investigation.get("cpu", {})
    error_section = investigation.get(
        "recent_errors",
        {},
    )

    health_status = health_section.get(
        "status",
        "unknown",
    )
    error_count = int(error_section.get("match_count", 0) or 0)

    cpu_unit = cpu_section.get("unit")
    cpu_field = "value" if cpu_unit in {"cpu_cores", "percent"} else "value"
    maximum_cpu = _maximum_numeric_value(
        cpu_section,
        cpu_field,
    )

    high_cpu = False

    if maximum_cpu is not None:
        if cpu_unit == "percent":
            high_cpu = maximum_cpu >= 80
        elif cpu_unit == "cpu_cores":
            high_cpu = maximum_cpu >= 0.8

    evidence = {
        "health_status": health_status,
        "recent_error_count": error_count,
        "maximum_cpu": maximum_cpu,
        "cpu_unit": cpu_unit,
    }

    if health_status == "down" and error_count > 0:
        diagnosis = (
            "The service is unreachable and recent error logs "
            "exist, suggesting a service or application failure."
        )
        recommended_checks = [
            "Inspect the newest error and traceback entries.",
            "Check the container status and restart history.",
            "Verify the service health endpoint.",
        ]

    elif health_status == "down":
        diagnosis = (
            "The service is unreachable but no recent error "
            "logs were found, suggesting a networking, DNS, "
            "scrape-target or configuration problem."
        )
        recommended_checks = [
            "Check Prometheus target status and lastError.",
            "Verify Docker service names and network connectivity.",
            "Confirm that the metrics endpoint is exposed.",
        ]

    elif health_status == "up" and high_cpu:
        diagnosis = (
            "The service is reachable but is experiencing high "
            "CPU usage, indicating performance pressure."
        )
        recommended_checks = [
            "Inspect request rate and latency.",
            "Check for expensive operations or load spikes.",
            "Review CPU limits and scaling configuration.",
        ]

    elif health_status == "up" and error_count > 0:
        diagnosis = (
            "The service is reachable, but recent application errors were found."
        )
        recommended_checks = [
            "Inspect recent error messages.",
            "Correlate errors with request and dependency metrics.",
            "Verify downstream services and credentials.",
        ]

    elif (
        health_status == "unknown"
        and error_count == 0
        and not cpu_section.get("values")
    ):
        diagnosis = (
            "There is insufficient metrics and log data to "
            "determine the service condition."
        )
        recommended_checks = [
            "Verify the Prometheus scrape configuration.",
            "Verify Fluent Bit log delivery to S3.",
            "Confirm the requested service name.",
        ]

    else:
        diagnosis = "The available evidence does not show an immediate service failure."
        recommended_checks = [
            "Continue monitoring health, CPU and recent errors.",
            "Check request rate and latency when investigating performance concerns.",
        ]

    return {
        "service": investigation["service"],
        "environment": investigation["environment"],
        "minutes": minutes,
        "diagnosis": diagnosis,
        "evidence": evidence,
        "recommended_checks": recommended_checks,
        "investigation": investigation,
    }


@mcp.tool
def investigate_incident(
    service: str,
    timestamp: str,
    environment: str = "dev",
    window_minutes: int = 5,
) -> dict[str, Any]:
    """Investigate logs and metrics around an incident timestamp."""
    normalized_service = _require_non_empty(
        "service",
        service,
    )
    normalized_environment = _normalize_environment(environment)

    _validate_int_range(
        "window_minutes",
        window_minutes,
        1,
        60,
    )

    incident_time = _parse_iso_timestamp(timestamp)
    start_time = incident_time - timedelta(minutes=window_minutes)
    end_time = incident_time + timedelta(minutes=window_minutes)

    escaped_service = _escape_promql_label_value(normalized_service)

    health_query = f'up{{job="{escaped_service}"}}'
    cpu_query = f'rate(process_cpu_seconds_total{{job="{escaped_service}"}}[5m])'

    step_seconds = max(
        15,
        int((end_time - start_time).total_seconds() / 40),
    )

    error_logs = _safe_section(
        lambda: _search_log_entries(
            environment=normalized_environment,
            keywords=ERROR_KEYWORDS,
            service=normalized_service,
            start_time=start_time,
            end_time=end_time,
            max_files=200,
            max_matches=200,
        )
    )

    health_metrics = _safe_section(
        lambda: _prometheus_range_query(
            health_query,
            normalized_environment,
            start_time=start_time,
            end_time=end_time,
            step_seconds=step_seconds,
        )
    )

    cpu_metrics = _safe_section(
        lambda: _prometheus_range_query(
            cpu_query,
            normalized_environment,
            start_time=start_time,
            end_time=end_time,
            step_seconds=step_seconds,
        )
    )

    historical_data_available = not (
        "error" in health_metrics and "error" in cpu_metrics
    )

    return {
        "service": normalized_service,
        "environment": normalized_environment,
        "incident_timestamp": incident_time.isoformat(),
        "window_minutes": window_minutes,
        "window_start": start_time.isoformat(),
        "window_end": end_time.isoformat(),
        "historical_metrics_available": (historical_data_available),
        "health_query": health_query,
        "cpu_query": cpu_query,
        "health_metrics": health_metrics,
        "cpu_metrics": cpu_metrics,
        "matching_error_logs": error_logs,
        "note": (
            "Historical metrics are limited by Prometheus "
            "retention and by whether the metric existed during "
            "the requested time window."
        ),
    }


if __name__ == "__main__":
    mcp.run()
