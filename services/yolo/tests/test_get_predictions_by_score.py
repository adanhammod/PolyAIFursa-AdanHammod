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
            "dog",
            0.42,
            "[50, 60, 150, 250]"
        ))

        conn.commit()


@pytest.fixture
def client():
    return TestClient(app)


def test_get_predictions_by_score_success(client):
    """
    Verify that the endpoint returns all detection objects
    whose score is greater than or equal to the requested score.
    """

    response = client.get("/predictions/score/0.5")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1

    assert data[0]["prediction_uid"] == "abc-123"
    assert data[0]["label"] == "person"
    assert data[0]["score"] == 0.91


def test_get_predictions_by_score_no_matches(client):
    """
    Verify that the endpoint returns an empty list when
    no detection objects satisfy the requested minimum score.
    """

    response = client.get("/predictions/score/0.99")

    assert response.status_code == 200
    assert response.json() == []


def test_get_predictions_by_score_invalid_range(client):
    """
    Verify that the endpoint returns HTTP 400 when
    min_score is outside the valid range [0.0, 1.0].
    """

    response = client.get("/predictions/score/1.5")

    assert response.status_code == 400

    assert response.json() == {
        "detail": "min_score must be between 0.0 and 1.0"
    }