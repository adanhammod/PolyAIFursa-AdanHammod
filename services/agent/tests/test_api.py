import base64
import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app as agent_app

client = TestClient(agent_app.app)

MOCK_AGENT_RETURN = ("Hello!", {"input": 10, "output": 5, "total": 15})


def test_chat_returns_response_and_tokens():
    with patch("app.run_agent", return_value=MOCK_AGENT_RETURN):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Hello!"
    assert data["tokens_used"] == {"input": 10, "output": 5, "total": 15}
    assert data["annotated_image_base64"] is None


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_with_image_uploads_to_s3_and_returns_annotated():
    fake_image_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 10
    image_b64 = base64.b64encode(fake_image_bytes).decode()

    annotated_bytes = b"fake-annotated-image"

    def fake_download(bucket, key, fileobj):
        fileobj.write(annotated_bytes)

    def fake_run_agent(_messages):
        agent_app._annotated_image_s3_key.set("predicted/yolo-uid-123.jpg")
        return (
            "I found 0 objects.",
            {"input": 11, "output": 6, "total": 17},
        )

    with patch.object(agent_app.s3_client, "download_fileobj") as mock_download:
        mock_download.side_effect = fake_download

        with patch("app.run_agent") as mock_run_agent:
            mock_run_agent.side_effect = fake_run_agent

            resp = client.post(
                "/chat",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is in this image?",
                            "image_base64": image_b64,
                        }
                    ]
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "I found 0 objects."
    assert data["tokens_used"] == {"input": 11, "output": 6, "total": 17}
    assert data["annotated_image_base64"] == base64.b64encode(annotated_bytes).decode()


def test_image_base64_not_forwarded_to_llm():
    """Raw base64 must not appear in messages passed to run_agent."""
    fake_b64 = base64.b64encode(b"FAKE_IMAGE_BYTES" * 20).decode()
    captured: list = []

    def spy(messages):
        captured.extend(messages)
        return ("ok", {"input": 1, "output": 1, "total": 2})

    with patch("app.run_agent", side_effect=spy):
        resp = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "What's in this image?",
                        "image_base64": fake_b64,
                    }
                ]
            },
        )

    assert resp.status_code == 200
    all_text = " ".join(
        m.content for m in captured
        if hasattr(m, "content") and isinstance(m.content, str)
    )
    assert fake_b64 not in all_text


def test_user_text_preserved_with_image():
    """User text must reach run_agent alongside the image marker."""
    fake_b64 = base64.b64encode(b"FAKE_IMAGE_BYTES" * 20).decode()
    captured: list = []

    def spy(messages):
        captured.extend(messages)
        return ("ok", {"input": 1, "output": 1, "total": 2})

    with patch("app.run_agent", side_effect=spy):
        resp = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "What's in this image?",
                        "image_base64": fake_b64,
                    }
                ]
            },
        )

    assert resp.status_code == 200
    human_msg = next(
        m for m in captured
        if isinstance(m, HumanMessage) and "[User uploaded an image.]" in m.content
    )
    assert "What's in this image?" in human_msg.content


def test_what_is_in_image_routes_to_detect_objects():
    """'What's in this image?' with an image upload should call detect_objects."""
    fake_b64 = base64.b64encode(b"img").decode()
    tool_response = MagicMock(spec=AIMessage)
    tool_response.content = ""
    tool_response.tool_calls = [{"name": "detect_objects", "id": "c1", "args": {}}]
    tool_response.usage_metadata = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}

    final_response = MagicMock(spec=AIMessage)
    final_response.content = "I found 2 cats."
    final_response.tool_calls = []
    final_response.usage_metadata = {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13}

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = ToolMessage(
        content='{"uid":"u1","annotated_image_s3_key":null,"detection_objects":[]}',
        tool_call_id="c1",
    )

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"detect_objects": mock_tool}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        resp = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "What's in this image?",
                        "image_base64": fake_b64,
                    }
                ]
            },
        )

    assert resp.status_code == 200
    mock_tool.invoke.assert_called_once()


def test_rotate_with_image_routes_to_rotate_tool():
    """'rotate the image 90' with an image upload should call the rotate MCP tool."""
    fake_b64 = base64.b64encode(b"img").decode()
    tool_response = MagicMock(spec=AIMessage)
    tool_response.content = ""
    tool_response.tool_calls = [{"name": "rotate", "id": "c2", "args": {"angle": 90.0}}]
    tool_response.usage_metadata = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}

    final_response = MagicMock(spec=AIMessage)
    final_response.content = "Rotated 90 degrees."
    final_response.tool_calls = []
    final_response.usage_metadata = {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}

    mock_rotate = MagicMock()
    mock_rotate.invoke.return_value = ToolMessage(
        content='{"processed_image_b64":"ZmFrZQ=="}',
        tool_call_id="c2",
    )
    mock_detect = MagicMock()

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"rotate": mock_rotate, "detect_objects": mock_detect}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        resp = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "rotate the image 90",
                        "image_base64": fake_b64,
                    }
                ]
            },
        )

    assert resp.status_code == 200
    mock_rotate.invoke.assert_called_once()
    mock_detect.invoke.assert_not_called()
    assert resp.json()["annotated_image_base64"] == "ZmFrZQ=="


def test_processed_image_returned_in_chat_response():
    """annotated_image_base64 in ChatResponse must carry the MCP-processed image."""
    fake_b64 = base64.b64encode(b"img").decode()
    processed_b64 = "cmVzdWx0X2ltYWdl"  # base64 for "result_image"

    tool_response = MagicMock(spec=AIMessage)
    tool_response.content = ""
    tool_response.tool_calls = [{"name": "blur", "id": "c3", "args": {"radius": 2.0}}]
    tool_response.usage_metadata = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}

    final_response = MagicMock(spec=AIMessage)
    final_response.content = "Blurred."
    final_response.tool_calls = []
    final_response.usage_metadata = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    mock_blur = MagicMock()
    mock_blur.invoke.return_value = ToolMessage(
        content=f'{{"processed_image_b64":"{processed_b64}"}}',
        tool_call_id="c3",
    )

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"blur": mock_blur}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        resp = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "blur the image",
                        "image_base64": fake_b64,
                    }
                ]
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["annotated_image_base64"] == processed_b64


def test_image_label_stripped_from_response_when_mcp_image_returned():
    """Standalone MCP labels ('Rotated image') must be stripped when an image is returned."""
    fake_b64 = base64.b64encode(b"img").decode()
    processed_b64 = "cmVzdWx0X2ltYWdl"

    tool_response = MagicMock(spec=AIMessage)
    tool_response.content = ""
    tool_response.tool_calls = [{"name": "rotate", "id": "c4", "args": {"angle": 90.0}}]
    tool_response.usage_metadata = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}

    final_response = MagicMock(spec=AIMessage)
    final_response.content = "I rotated the image 90 degrees.\nRotated image"
    final_response.tool_calls = []
    final_response.usage_metadata = {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}

    mock_rotate = MagicMock()
    mock_rotate.invoke.return_value = ToolMessage(
        content=f'{{"processed_image_b64":"{processed_b64}"}}',
        tool_call_id="c4",
    )

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"rotate": mock_rotate}),
    ):
        mock_llm.invoke.side_effect = [tool_response, final_response]
        resp = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "rotate the image 90",
                        "image_base64": fake_b64,
                    }
                ]
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "Rotated image" not in data["response"]
    assert "I rotated the image 90 degrees." in data["response"]
    assert data["annotated_image_base64"] == processed_b64


def test_frontend_response_never_contains_reasoning_content():
    """The /chat response field must never expose raw reasoning_content to the frontend."""
    llm_content = [
        {"type": "reasoning_content", "reasoning_content": {"text": "secret reasoning"}},
        {"type": "text", "text": "Here is your answer."},
    ]

    def fake_run_agent(_messages):
        return (agent_app._extract_visible_text(llm_content), {"input": 1, "output": 1, "total": 2})

    with patch("app.run_agent", side_effect=fake_run_agent):
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "reasoning_content" not in data["response"]
    assert "secret reasoning" not in data["response"]
    assert data["response"] == "Here is your answer."


def test_consecutive_requests_use_different_images():
    """Two consecutive /chat requests with different images must each use their own image."""
    imageA = base64.b64encode(b"imageA").decode()
    imageB = base64.b64encode(b"imageB").decode()
    captured: list[str | None] = []

    def spy(messages):
        captured.append(agent_app._current_image_b64.get())
        return ("done", {"input": 1, "output": 1, "total": 2})

    with patch("app.run_agent", side_effect=spy):
        # Request 1: single user message with imageA
        client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "blur", "image_base64": imageA}]},
        )
        # Request 2: history includes old user msg with imageA; newest user msg has imageB
        client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "blur", "image_base64": imageA},
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "rotate", "image_base64": imageB},
                ]
            },
        )

    assert captured[0] == imageA, "Request 1 must use imageA"
    assert captured[1] == imageB, "Request 2 must use imageB, not old imageA"
    assert captured[0] != captured[1]


def test_new_request_without_upload_does_not_fall_back_to_old_image():
    """If the newest user message has no image_base64, _current_image_b64 must be None.

    The backend must not fall back to image_base64 from an older user message in
    the conversation history — that would silently process the wrong image.
    """
    imageA = base64.b64encode(b"imageA").decode()
    captured: list[str | None] = []

    def spy(messages):
        captured.append(agent_app._current_image_b64.get())
        return ("done", {"input": 1, "output": 1, "total": 2})

    with patch("app.run_agent", side_effect=spy):
        # History has imageA in an old user message; newest user message has NO image
        client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "blur", "image_base64": imageA},
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "what did you do?"},  # no image
                ]
            },
        )

    assert captured[0] is None, (
        "_current_image_b64 must be None when latest user message has no upload; "
        "must not fall back to old imageA"
    )


# ---------------------------------------------------------------------------
# Helpers for tests that exercise the MCP code path via mock_call_mcp
# ---------------------------------------------------------------------------

def _ai_tool_call(tool_name: str, call_id: str) -> MagicMock:
    msg = MagicMock(spec=AIMessage)
    msg.content = ""
    msg.tool_calls = [{"name": tool_name, "id": call_id, "args": {}}]
    msg.usage_metadata = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
    return msg


def _ai_final(text: str) -> MagicMock:
    msg = MagicMock(spec=AIMessage)
    msg.content = text
    msg.tool_calls = []
    msg.usage_metadata = {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}
    return msg


def _make_mcp_tool(tool_name: str, call_mcp) -> MagicMock:
    """Mock tool whose invoke reads _current_image_b64 and delegates to call_mcp.

    This mirrors what the real _build_image_proc_wrapper closure does, without
    needing TOOLS to be pre-populated by the lifespan MCP discovery step.
    """
    mock = MagicMock()

    def invoke(tool_call):
        image_b64 = agent_app._current_image_b64.get()
        result = call_mcp(tool_name, {"image_b64": image_b64})
        return ToolMessage(
            content=json.dumps({"processed_image_b64": result}),
            tool_call_id=tool_call["id"],
        )

    mock.invoke.side_effect = invoke
    return mock


def test_consecutive_mcp_calls_use_different_images(mock_call_mcp):
    """Requirement 1: each /chat request must forward its own image to _call_mcp."""
    imageA = base64.b64encode(b"IMAGE_A").decode()
    imageB = base64.b64encode(b"IMAGE_B").decode()
    processedA = base64.b64encode(b"PROCESSED_A").decode()
    processedB = base64.b64encode(b"PROCESSED_B").decode()

    mcp_inputs = []

    def tracking(tool_name, arguments):
        mcp_inputs.append(arguments.get("image_b64"))
        return processedA if arguments.get("image_b64") == imageA else processedB

    mock_call_mcp.side_effect = tracking
    blur = _make_mcp_tool("blur", mock_call_mcp)
    rotate = _make_mcp_tool("rotate", mock_call_mcp)

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"blur": blur, "rotate": rotate}),
    ):
        mock_llm.invoke.side_effect = [
            _ai_tool_call("blur", "c1"), _ai_final("Done! I blurred the image."),
            _ai_tool_call("rotate", "c2"), _ai_final("Done! I rotated the image 90° clockwise."),
        ]
        client.post("/chat", json={"messages": [
            {"role": "user", "content": "blur", "image_base64": imageA}
        ]})
        client.post("/chat", json={"messages": [
            {"role": "user", "content": "rotate", "image_base64": imageB}
        ]})

    assert len(mcp_inputs) == 2
    assert mcp_inputs[0] == imageA, "Request 1 must pass imageA to MCP"
    assert mcp_inputs[1] == imageB, "Request 2 must pass imageB to MCP"
    assert mcp_inputs[0] != mcp_inputs[1]


def test_second_request_response_excludes_first_response_text(mock_call_mcp):
    """Requirement 2: second response must not echo the first response's text."""
    imageA = base64.b64encode(b"IMAGE_A").decode()
    imageB = base64.b64encode(b"IMAGE_B").decode()
    mock_call_mcp.return_value = base64.b64encode(b"PROCESSED").decode()
    blur = _make_mcp_tool("blur", mock_call_mcp)
    rotate = _make_mcp_tool("rotate", mock_call_mcp)

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"blur": blur, "rotate": rotate}),
    ):
        mock_llm.invoke.side_effect = [
            _ai_tool_call("blur", "c1"), _ai_final("Done! I blurred the image."),
            _ai_tool_call("rotate", "c2"), _ai_final("Done! I rotated the image 90° clockwise."),
        ]
        client.post("/chat", json={"messages": [
            {"role": "user", "content": "blur", "image_base64": imageA}
        ]})
        resp2 = client.post("/chat", json={"messages": [
            {"role": "user", "content": "rotate", "image_base64": imageB}
        ]})

    data2 = resp2.json()
    assert "blurred" not in data2["response"].lower(), (
        "Request 2 response must not contain text from request 1"
    )
    assert "rotated" in data2["response"].lower()


def test_second_request_annotated_image_differs_from_first(mock_call_mcp):
    """Requirement 3: annotated_image_base64 from request 2 must come from imageB."""
    imageA = base64.b64encode(b"IMAGE_A").decode()
    imageB = base64.b64encode(b"IMAGE_B").decode()
    processedA = base64.b64encode(b"PROCESSED_A").decode()
    processedB = base64.b64encode(b"PROCESSED_B").decode()

    def tracking(tool_name, arguments):
        return processedA if arguments.get("image_b64") == imageA else processedB

    mock_call_mcp.side_effect = tracking
    blur = _make_mcp_tool("blur", mock_call_mcp)
    rotate = _make_mcp_tool("rotate", mock_call_mcp)

    with (
        patch.object(agent_app, "llm_with_tools") as mock_llm,
        patch.dict(agent_app.TOOLS, {"blur": blur, "rotate": rotate}),
    ):
        mock_llm.invoke.side_effect = [
            _ai_tool_call("blur", "c1"), _ai_final("Done! I blurred the image."),
            _ai_tool_call("rotate", "c2"), _ai_final("Done! I rotated the image 90° clockwise."),
        ]
        resp1 = client.post("/chat", json={"messages": [
            {"role": "user", "content": "blur", "image_base64": imageA}
        ]})
        resp2 = client.post("/chat", json={"messages": [
            {"role": "user", "content": "rotate", "image_base64": imageB}
        ]})

    assert resp1.json()["annotated_image_base64"] == processedA
    assert resp2.json()["annotated_image_base64"] == processedB
    assert resp1.json()["annotated_image_base64"] != resp2.json()["annotated_image_base64"]
