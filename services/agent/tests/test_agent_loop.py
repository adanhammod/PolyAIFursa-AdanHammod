import asyncio
import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app as agent_app

_BEDROCK_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


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


def test_rotate_tool_call():
    """MCP-discovered 'rotate' tool (not the old 'rotate_image' wrapper)."""
    tool_response = _ai_msg(
        content="",
        tool_calls=[{"name": "rotate", "id": "call_rot", "args": {"angle": 90.0}}],
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
        patch.dict(agent_app.TOOLS, {"rotate": mock_tool}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        text, tokens = agent_app.run_agent(
            [HumanMessage(content="rotate the image 90 degrees")]
        )

    assert text == "Image has been rotated 90 degrees."
    assert tokens == {"input": 25, "output": 9, "total": 34}
    assert mock_llm.invoke.call_count == 2
    assert mock_tool.invoke.call_count == 1


def test_rotate_request_calls_rotate_not_detect_objects():
    """'rotate the image 90' must route to 'rotate', never to 'detect_objects'."""
    tool_response = _ai_msg(
        content="",
        tool_calls=[{"name": "rotate", "id": "call_r", "args": {"angle": 90.0}}],
        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )
    final_response = _ai_msg(
        "Rotated.",
        usage={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
    )

    mock_rotate = MagicMock()
    mock_rotate.invoke.return_value = _fake_tool_result(
        {"processed_image_b64": "ZmFrZQ=="}, call_id="call_r"
    )
    mock_detect = MagicMock()

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"rotate": mock_rotate, "detect_objects": mock_detect}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        agent_app.run_agent([HumanMessage(content="rotate the image 90")])

    mock_rotate.invoke.assert_called_once()
    mock_detect.invoke.assert_not_called()


def test_blur_request_calls_blur_not_detect_objects():
    """'blur the image' must route to 'blur', never to 'detect_objects'."""
    tool_response = _ai_msg(
        content="",
        tool_calls=[{"name": "blur", "id": "call_b", "args": {"radius": 2.0}}],
        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )
    final_response = _ai_msg(
        "Blurred.",
        usage={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
    )

    mock_blur = MagicMock()
    mock_blur.invoke.return_value = _fake_tool_result(
        {"processed_image_b64": "ZmFrZQ=="}, call_id="call_b"
    )
    mock_detect = MagicMock()

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"blur": mock_blur, "detect_objects": mock_detect}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        agent_app.run_agent([HumanMessage(content="blur the image")])

    mock_blur.invoke.assert_called_once()
    mock_detect.invoke.assert_not_called()


def test_processed_b64_stripped_from_llm_messages():
    """processed_image_b64 must not appear in any messages sent to llm_with_tools.invoke()."""
    large_b64 = "A" * 500  # simulates a large processed image
    tool_response = _ai_msg(
        content="",
        tool_calls=[{"name": "blur", "id": "call_b", "args": {"radius": 2.0}}],
        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )
    final_response = _ai_msg(
        "Blurred.",
        usage={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
    )
    mock_blur = MagicMock()
    mock_blur.invoke.return_value = _fake_tool_result(
        {"processed_image_b64": large_b64}, call_id="call_b"
    )

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"blur": mock_blur}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        text, _ = agent_app.run_agent([HumanMessage(content="blur the image")])

    assert text == "Blurred."
    assert mock_llm.invoke.call_count == 2

    # Verify large_b64 absent from every message in every invoke call
    all_content = " ".join(
        m.content
        for call in mock_llm.invoke.call_args_list
        for m in call.args[0]
        if hasattr(m, "content") and isinstance(m.content, str)
    )
    assert large_b64 not in all_content
    assert '"processed_image": true' in all_content.lower() or "processed_image" in all_content


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


# ---------------------------------------------------------------------------
# MCP discovery tests
# ---------------------------------------------------------------------------

def _run_init_tools_with_mock(fake_tools):
    """Run _init_tools() in a fresh event loop with MultiServerMCPClient mocked."""
    mock_instance = MagicMock()
    mock_instance.get_tools = AsyncMock(return_value=fake_tools)
    mock_cls = MagicMock(return_value=mock_instance)

    with patch.object(agent_app, "MultiServerMCPClient", mock_cls):
        asyncio.run(agent_app._init_tools())


def _make_fake_mcp_tool(name: str, extra_properties: dict | None = None):
    """Build a fake MCP tool using the JSON Schema dict format that
    langchain-mcp-adapters v0.3+ actually returns."""
    t = MagicMock()
    t.name = name
    t.description = f"Apply {name} to the image."
    properties = {"image_b64": {"type": "string"}}
    if extra_properties:
        properties.update(extra_properties)
    t.args_schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }
    return t


def test_mcp_tools_are_discovered():
    """After _init_tools(), TOOLS contains all MCP-discovered tool names."""
    fake_tools = [_make_fake_mcp_tool(n) for n in ["rotate", "flip", "blur", "resize", "crop", "add_noise"]]
    _run_init_tools_with_mock(fake_tools)
    expected = {"rotate", "flip", "blur", "resize", "crop", "add_noise"}
    assert expected.issubset(agent_app.TOOLS.keys())


def test_bind_tools_receives_detect_objects_plus_mcp_tools():
    """llm.bind_tools is called with detect_objects + all MCP tools."""
    fake_tools = [_make_fake_mcp_tool(n) for n in ["blur", "add_noise"]]
    _run_init_tools_with_mock(fake_tools)
    call_args = agent_app.llm.bind_tools.call_args
    assert call_args is not None, "llm.bind_tools was never called"
    bound_tools = call_args[0][0]
    names = {t.name for t in bound_tools}
    assert "detect_objects" in names
    assert {"blur", "add_noise"}.issubset(names)


def test_blur_and_add_noise_visible_to_llm():
    """blur and add_noise appear in TOOLS after discovery."""
    fake_tools = [_make_fake_mcp_tool(n) for n in ["blur", "resize", "crop", "add_noise"]]
    _run_init_tools_with_mock(fake_tools)
    assert "blur" in agent_app.TOOLS
    assert "add_noise" in agent_app.TOOLS


def test_detect_objects_still_present_after_discovery():
    """detect_objects is always in TOOLS regardless of what MCP returns."""
    fake_tools = [_make_fake_mcp_tool("blur")]
    _run_init_tools_with_mock(fake_tools)
    assert "detect_objects" in agent_app.TOOLS
    assert agent_app.TOOLS["detect_objects"] is agent_app.detect_objects


# ---------------------------------------------------------------------------
# Tool name sanitization tests
# ---------------------------------------------------------------------------

def test_sanitize_tool_name_clean():
    """Valid names are returned unchanged."""
    assert agent_app._sanitize_tool_name("rotate") == "rotate"
    assert agent_app._sanitize_tool_name("add_noise") == "add_noise"
    assert agent_app._sanitize_tool_name("detect_objects") == "detect_objects"


def test_sanitize_tool_name_hyphens():
    """Hyphens are replaced with underscores."""
    assert agent_app._sanitize_tool_name("img-proc_blur") == "img_proc_blur"
    assert agent_app._sanitize_tool_name("some-tool") == "some_tool"


def test_sanitize_tool_name_leading_digit():
    """Names starting with a digit get a 't_' prefix."""
    assert agent_app._sanitize_tool_name("123tool").startswith("t_")


def test_hyphenated_mcp_name_sanitized_in_tools():
    """If MCP returns a hyphenated tool name, TOOLS uses the sanitized version."""
    fake_tool = _make_fake_mcp_tool("img-proc_blur")
    _run_init_tools_with_mock([fake_tool])
    assert "img_proc_blur" in agent_app.TOOLS
    assert "img-proc_blur" not in agent_app.TOOLS


def test_sanitized_tool_still_calls_mcp_with_original_name(monkeypatch):
    """The wrapper invokes _call_mcp with the original MCP tool name, not the sanitized name."""
    calls = []
    monkeypatch.setattr(agent_app, "_call_mcp", lambda name, args: calls.append(name) or "ZmFrZQ==")
    monkeypatch.setattr(agent_app, "_current_image_b64", type(agent_app._current_image_b64)("_current_image_b64", default="dGVzdA=="))

    fake_tool = _make_fake_mcp_tool("img-proc_blur")
    _run_init_tools_with_mock([fake_tool])

    wrapper = agent_app.TOOLS.get("img_proc_blur")
    assert wrapper is not None, "sanitized name not in TOOLS"
    # Invoke the underlying function directly to bypass ContextVar issues
    agent_app._current_image_b64.set("dGVzdA==")
    wrapper.func()
    assert calls == ["img-proc_blur"]


# ---------------------------------------------------------------------------
# Bedrock toolUse.name validation — regression tests
# ---------------------------------------------------------------------------

def test_all_bound_tool_names_valid_for_bedrock():
    """Every name in TOOLS and every name passed to bind_tools must match [a-zA-Z0-9_-]+."""
    fake_tools = [_make_fake_mcp_tool(n) for n in ["rotate", "flip", "blur", "resize", "crop", "add_noise"]]
    _run_init_tools_with_mock(fake_tools)

    for name in agent_app.TOOLS:
        assert _BEDROCK_NAME_RE.match(name), f"TOOLS key {name!r} invalid for Bedrock"

    call_args = agent_app.llm.bind_tools.call_args
    assert call_args is not None
    for t in call_args[0][0]:
        assert _BEDROCK_NAME_RE.match(t.name), f"Bound tool {t.name!r} invalid for Bedrock"


def test_garbled_tool_name_sanitized_before_history():
    """'rotate<|channel|>commentary' must be truncated to 'rotate' before AIMessage enters history."""
    garbled = "rotate<|channel|>commentary"
    tool_response = _ai_msg(
        content="",
        tool_calls=[{"name": garbled, "id": "call_r", "args": {"angle": 90.0}, "type": "tool_call"}],
        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )
    final_response = _ai_msg("Rotated.", usage={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10})

    mock_rotate = MagicMock()
    mock_rotate.invoke.return_value = _fake_tool_result({"processed_image_b64": "ZmFrZQ=="}, call_id="call_r")

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"rotate": mock_rotate}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        text, _ = agent_app.run_agent([HumanMessage(content="rotate the image")])

    assert text == "Rotated."
    mock_rotate.invoke.assert_called_once()

    # Every toolUse.name in every AIMessage sent to the second invoke must be valid.
    second_call_messages = mock_llm.invoke.call_args_list[1].args[0]
    for msg in second_call_messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            assert _BEDROCK_NAME_RE.match(tc["name"]), (
                f"Invalid toolUse.name {tc['name']!r} would cause Bedrock ValidationException"
            )


def test_unrecoverable_tool_name_dropped_no_key_error():
    """A name with no valid prefix (e.g. '<invalid>') must be dropped without raising KeyError."""
    tool_response = _ai_msg(
        content="fallback text",
        tool_calls=[{"name": "<invalid>", "id": "call_x", "args": {}, "type": "tool_call"}],
        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )

    with patch.object(agent_app, "llm_with_tools") as mock_llm:
        mock_llm.invoke.return_value = tool_response
        text, _ = agent_app.run_agent([HumanMessage(content="do something")])

    # The bad call is dropped; the agent returns the text content without crashing.
    assert mock_llm.invoke.call_count == 1
    assert text == "fallback text"


def test_unknown_tool_name_gets_error_tool_message():
    """A valid-format but unknown tool name must receive an error ToolMessage, not a KeyError."""
    tool_response = _ai_msg(
        content="",
        tool_calls=[{"name": "nonexistent_tool", "id": "call_y", "args": {}, "type": "tool_call"}],
        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )
    final_response = _ai_msg("Sorry.", usage={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10})

    with patch.object(agent_app, "llm_with_tools") as mock_llm:
        mock_llm.invoke.side_effect = [tool_response, final_response]
        text, _ = agent_app.run_agent([HumanMessage(content="do something")])

    assert text == "Sorry."
    # Second call must include a ToolMessage with an error for the unknown tool
    second_call_messages = mock_llm.invoke.call_args_list[1].args[0]
    tool_msgs = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0].content)
    assert "error" in payload
    assert "nonexistent_tool" in payload["error"]
