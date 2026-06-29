import os
from unittest.mock import MagicMock, patch

# Must be set before app.py is imported (module-level guard raises SystemExit if unset)
os.environ.setdefault("MODEL", "openai:gpt-5.4-mini")

_mock_llm = MagicMock()
_mock_llm.bind_tools.return_value = MagicMock()

# Patch init_chat_model BEFORE importing app so the module-level call receives the mock.
# Python caches the import in sys.modules, so all test files share the same patched module.
with patch("app.init_chat_model", return_value=_mock_llm):
    import app  # noqa: E402, F401
