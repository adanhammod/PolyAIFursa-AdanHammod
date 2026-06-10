import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app, init_db
from contextlib import closing

@pytest.fixture
def client():
    temp_dir = tempfile.TemporaryDirectory()

    app_module.DB_PATH = os.path.join(temp_dir.name, "test.db")
    init_db()

    with closing(sqlite3.connect(app_module.DB_PATH)) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions
            (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (
            "abc-123",
            "original.jpg",
            "predicted.jpg"
        ))

        conn.execute("""
            INSERT INTO detection_objects
            (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (
            "abc-123",
            "person",
            0.91,
            "[10, 20, 100, 200]"
        ))
        conn.commit()

    with TestClient(app) as test_client:
        yield test_client

    temp_dir.cleanup()


def test_get_prediction_by_uid_returns_prediction_with_objects(client):
    response = client.get("/prediction/abc-123")

    assert response.status_code == 200

    body = response.json()

    assert body["uid"] == "abc-123"
    assert body["original_image"] == "original.jpg"
    assert body["predicted_image"] == "predicted.jpg"

    assert len(body["detection_objects"]) == 1
    assert body["detection_objects"][0]["label"] == "person"
    assert body["detection_objects"][0]["score"] == 0.91
    assert body["detection_objects"][0]["box"] == "[10, 20, 100, 200]"


def test_get_prediction_by_uid_returns_404_when_uid_not_found(client):
    response = client.get("/prediction/not-found")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"