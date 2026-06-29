import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("MODEL", "openai:gpt-5.4-mini")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")


_mock_llm = MagicMock()
_mock_llm.profile = {"tool_calling": True}
_mock_llm.bind_tools.return_value = MagicMock()

with patch("langchain.chat_models.init_chat_model", return_value=_mock_llm):
    with patch("langchain_openai.ChatOpenAI", MagicMock()):
        import app  # noqa: E402
