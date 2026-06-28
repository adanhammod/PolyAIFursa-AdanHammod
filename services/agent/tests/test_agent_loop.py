from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app as agent_app


def _ai_msg(content, tool_calls=None, usage=None):
    """Build a minimal AIMessage substitute with controllable fields."""
    msg = MagicMock(spec=AIMessage)
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.usage_metadata = usage or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    return msg


def test_plain_text_response():
    ai_msg = _ai_msg(
        "Hi there!",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    with patch.object(agent_app, "llm_with_tools") as mock_llm:
        mock_llm.invoke.return_value = ai_msg
        text, tokens = agent_app.run_agent([HumanMessage(content="hello")])

    assert text == "Hi there!"
    assert tokens == {"input": 10, "output": 5, "total": 15}


def test_tool_call_then_text():
    tool_response = _ai_msg(
        content="",
        tool_calls=[{"name": "detect_objects", "id": "call_1", "args": {}}],
        usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )
    final_response = _ai_msg(
        "I found 2 objects.",
        usage={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20},
    )

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = ToolMessage(
        content='{"uid": "u1", "annotated_image_s3_key": "predicted/u1.jpg", "detection_objects": []}',
        tool_call_id="call_1",
    )

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"detect_objects": mock_tool}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        text, tokens = agent_app.run_agent(
            [HumanMessage(content="what's in this image?")]
        )

    assert text == "I found 2 objects."
    assert tokens == {"input": 25, "output": 8, "total": 33}


def test_max_iterations_exceeded():
    looping_response = _ai_msg(
        content="",
        tool_calls=[{"name": "detect_objects", "id": "call_x", "args": {}}],
        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = ToolMessage(content="{}", tool_call_id="call_x")

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"detect_objects": mock_tool}),
    ):
        mock_llm.invoke.return_value = looping_response
        text, tokens = agent_app.run_agent(
            [HumanMessage(content="hello")], max_iterations=2
        )

    assert "Error" in text
    assert "maximum iterations" in text
