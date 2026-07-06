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
