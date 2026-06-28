import base64
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app as app_module
from app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_no_image_returns_text(client, mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="No image provided.", tool_calls=[])

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert "No image provided." in data["response"]
    assert data["annotated_image_base64"] is None


def test_chat_with_image_uploads_to_s3_and_returns_annotated(client, mock_llm, mock_s3):
    fake_image_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 10  # minimal JPEG header
    image_b64 = base64.b64encode(fake_image_bytes).decode()

    yolo_result = {
        "uid": "yolo-uid-123",
        "annotated_image_s3_key": "predicted/yolo-uid-123.jpg",
        "detection_objects": [],
        "processing_time_s": 0.1,
    }

    first_response = AIMessage(
        content="",
        tool_calls=[{"name": "detect_objects", "id": "call-1", "args": {}}],
    )
    second_response = AIMessage(content="I found 0 objects.", tool_calls=[])

    call_count = 0

    def fake_invoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return first_response
        return second_response

    mock_llm.invoke.side_effect = fake_invoke

    annotated_bytes = b"fake-annotated-image"

    def fake_download(bucket, key, fileobj):
        fileobj.write(annotated_bytes)

    mock_s3.download_fileobj.side_effect = fake_download

    with pytest.MonkeyPatch.context() as mp:
        fake_httpx_response = MagicMock()
        fake_httpx_response.raise_for_status = MagicMock()
        fake_httpx_response.json.return_value = yolo_result

        import httpx

        original_post = None

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, **kwargs):
                return fake_httpx_response

        mp.setattr("app.httpx.Client", FakeClient)

        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "What is in this image?", "image_base64": image_b64}
                ]
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert mock_s3.upload_fileobj.called
    assert data["annotated_image_base64"] == base64.b64encode(annotated_bytes).decode()
