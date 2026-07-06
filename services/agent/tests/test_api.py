import base64
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
