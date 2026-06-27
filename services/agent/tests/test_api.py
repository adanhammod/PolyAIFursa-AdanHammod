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
    assert "response" in data
    assert "tokens_used" in data


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
