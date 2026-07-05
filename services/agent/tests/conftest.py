import os
import sys
from unittest.mock import MagicMock, patch

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

with patch("langchain.chat_models.init_chat_model", return_value=_mock_llm):
    import app  # noqa: E402


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
