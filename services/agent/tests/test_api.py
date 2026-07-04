import base64
from unittest.mock import patch

from fastapi.testclient import TestClient

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
