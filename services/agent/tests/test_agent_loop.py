import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app as agent_app


def _ai_msg(content="", tool_calls=None, usage=None):
    msg = MagicMock(spec=AIMessage)
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.usage_metadata = usage or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    return msg


def _fake_tool_result(payload: dict, call_id="call_1"):
    return ToolMessage(
        content=json.dumps(payload),
        tool_call_id=call_id,
    )


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
    assert mock_llm.invoke.call_count == 1


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
    mock_tool.invoke.return_value = _fake_tool_result(
        {
            "uid": "u1",
            "annotated_image_s3_key": "predicted/u1.jpg",
            "detection_objects": [],
        },
        call_id="call_1",
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
    assert mock_llm.invoke.call_count == 2
    assert mock_tool.invoke.call_count == 1


def test_rotate_image_tool_call():
    tool_response = _ai_msg(
        content="",
        tool_calls=[{"name": "rotate_image", "id": "call_rot", "args": {"angle": 90.0}}],
        usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )
    final_response = _ai_msg(
        "Image has been rotated 90 degrees.",
        usage={"input_tokens": 15, "output_tokens": 6, "total_tokens": 21},
    )

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = _fake_tool_result(
        {"processed_image_b64": "ZmFrZWltYWdl"},
        call_id="call_rot",
    )

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"rotate_image": mock_tool}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        text, tokens = agent_app.run_agent(
            [HumanMessage(content="rotate the image 90 degrees")]
        )

    assert text == "Image has been rotated 90 degrees."
    assert tokens == {"input": 25, "output": 9, "total": 34}
    assert mock_llm.invoke.call_count == 2
    assert mock_tool.invoke.call_count == 1


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
            [HumanMessage(content="hello")],
            max_iterations=2,
        )

    assert "maximum iterations" in text
    assert tokens == {"input": 10, "output": 4, "total": 14}
    assert mock_llm.invoke.call_count == 2
