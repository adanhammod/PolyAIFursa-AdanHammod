import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault(
    "MODEL",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
)
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

# Ensure `import app` resolves to services/agent/app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_mock_llm = MagicMock()
_mock_llm.profile = {"tool_calling": True}
_mock_llm.bind_tools.return_value = MagicMock()

# Fake MCP tools that mirror the real img-proc-mcp server's tool names.
_FAKE_MCP_TOOL_NAMES = ["rotate", "flip", "blur", "resize", "crop", "add_noise"]


def _make_fake_mcp_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = f"Apply {name} to the user's image."
    schema = MagicMock()
    # No extra fields beyond image_b64; the wrapper strips image_b64, leaving
    # an empty schema (which is fine — the LLM just calls the tool with no args).
    schema.model_fields = {}
    t.args_schema = schema
    return t


_fake_mcp_tools = [_make_fake_mcp_tool(n) for n in _FAKE_MCP_TOOL_NAMES]

_mock_mcp_instance = MagicMock()
_mock_mcp_instance.get_tools = AsyncMock(return_value=_fake_mcp_tools)
_mock_mcp_client_cls = MagicMock(return_value=_mock_mcp_instance)

# Patch init_chat_model so app.py never calls AWS on import.
_llm_patcher = patch("langchain.chat_models.init_chat_model", return_value=_mock_llm)
_llm_patcher.start()

import app  # noqa: E402

# Replace app.MultiServerMCPClient with the mock for the entire test session so
# that the FastAPI lifespan (_init_tools) never dials a real MCP server.
_mcp_patcher = patch.object(app, "MultiServerMCPClient", _mock_mcp_client_cls)
_mcp_patcher.start()


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr("app.llm_with_tools", fake)
    return fake


@pytest.fixture(autouse=True)
def mock_s3(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr("app.s3_client", fake)
    return fake


@pytest.fixture
def mock_call_mcp(monkeypatch):
    """Patch _call_mcp so tests don't need a live MCP server."""
    fake = MagicMock(return_value="ZmFrZWltYWdl")  # dummy base64 string
    monkeypatch.setattr("app._call_mcp", fake)
    return fake
