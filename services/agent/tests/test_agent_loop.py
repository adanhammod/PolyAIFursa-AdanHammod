import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app as app_module
from app import run_agent


def _ai_text(text):
    return AIMessage(content=text, tool_calls=[])


def _ai_tool_call(call_id="call-1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "detect_objects", "id": call_id, "args": {}}],
    )


def _fake_tool_result(payload: dict) -> MagicMock:
    result = MagicMock(spec=ToolMessage)
    result.content = json.dumps(payload)
    return result


def test_run_agent_plain_text_response(mock_llm):
    mock_llm.invoke.return_value = _ai_text("Hello from agent.")

    result = run_agent([HumanMessage(content="Hi")])

    assert result == "Hello from agent."
    assert mock_llm.invoke.call_count == 1


def test_run_agent_tool_call_then_text(mock_llm, monkeypatch):
    tool_result = _fake_tool_result({
        "uid": "u1",
        "annotated_image_s3_key": "predicted/u1.jpg",
        "detection_objects": [],
    })

    call_count = 0

    def fake_invoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _ai_tool_call()
        return _ai_text("Done processing.")

    mock_llm.invoke.side_effect = fake_invoke

    fake_tool = MagicMock()
    fake_tool.invoke.return_value = tool_result
    monkeypatch.setattr("app.TOOLS", {"detect_objects": fake_tool})

    result = run_agent([HumanMessage(content="Analyze this.")])

    assert result == "Done processing."
    assert mock_llm.invoke.call_count == 2
    assert fake_tool.invoke.call_count == 1


def test_run_agent_max_iterations_exceeded(mock_llm, monkeypatch):
    mock_llm.invoke.return_value = _ai_tool_call()

    fake_tool = MagicMock()
    fake_tool.invoke.return_value = _fake_tool_result({})
    monkeypatch.setattr("app.TOOLS", {"detect_objects": fake_tool})

    result = run_agent([HumanMessage(content="Loop forever.")], max_iterations=3)

    assert "exceeded maximum iterations" in result
    assert mock_llm.invoke.call_count == 3
