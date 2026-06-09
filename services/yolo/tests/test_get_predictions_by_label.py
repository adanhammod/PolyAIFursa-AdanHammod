import os
import pytest
from fastapi.testclient import TestClient
import sqlite3
from contextlib import closing

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

from app import app, init_db

@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_predictions.db")

    monkeypatch.setattr("app.DB_PATH", db_file)

    init_db()

    with closing(sqlite3.connect(db_file)) as conn:
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

        conn.execute("""
            INSERT INTO detection_objects
            (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (
            "abc-123",
            "person",
            0.87,
            "[50, 60, 150, 250]"
        ))

        conn.commit()


@pytest.fixture
def client():
    return TestClient(app)


def test_get_predictions_by_existing_label(client):
    """
    Verify that the endpoint returns all prediction sessions
    containing the requested label and includes the matching
    detection objects in the response.
    """
    response = client.get("/predictions/label/person")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["uid"] == "abc-123"

    assert len(data[0]["detection_objects"]) == 2

    assert data[0]["detection_objects"][0]["label"] == "person"
    assert data[0]["detection_objects"][1]["label"] == "person"


def test_get_predictions_by_label_not_found(client):
    """
    Verify that the endpoint returns an empty list when
    no prediction sessions contain the requested label.
    """
    response = client.get("/predictions/label/car")

    assert response.status_code == 200
    assert response.json() == []


def test_get_predictions_by_empty_label(client):
    """
    Verify that the endpoint returns HTTP 400 when the
    provided label is empty or contains only whitespace.
    """
    response = client.get("/predictions/label/%20")

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Label cannot be empty"
    }