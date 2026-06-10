import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient
from contextlib import closing

import app as app_module
from app import app, init_db

@pytest.fixture
def client_with_image():
    temp_dir = tempfile.TemporaryDirectory()

    app_module.DB_PATH = os.path.join(temp_dir.name, "test.db")
    init_db()

    image_path = os.path.join(temp_dir.name, "predicted.jpg")

    with open(image_path, "wb") as f:
        f.write(b"fake image content")

    with closing(sqlite3.connect(app_module.DB_PATH)) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions
            (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (
            "abc-123",
            "original.jpg",
            image_path
        ))
        conn.commit()

    with TestClient(app) as client:
        yield client

    temp_dir.cleanup()


def test_get_prediction_image_returns_file_when_exists(client_with_image):
    response = client_with_image.get("/prediction/abc-123/image")

    assert response.status_code == 200
    assert response.content == b"fake image content"


def test_get_prediction_image_returns_404_when_uid_not_found(client_with_image):
    response = client_with_image.get("/prediction/not-found/image")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_get_prediction_image_returns_404_when_file_missing(client_with_image):
    with closing(sqlite3.connect(app_module.DB_PATH)) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions
            (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (
            "missing-file",
            "original.jpg",
            "not_existing_file.jpg"
        ))
        conn.commit()   



    response = client_with_image.get("/prediction/missing-file/image")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"