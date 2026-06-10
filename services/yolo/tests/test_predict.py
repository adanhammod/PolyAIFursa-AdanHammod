import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

from app import app, init_db


class FakeBox:
    cls = [type("FakeValue", (), {"item": lambda self: 0})()]
    conf = [0.91]
    xyxy = [type("FakeXYXY", (), {"tolist": lambda self: [10, 20, 100, 200]})()]


class FakeResult:
    boxes = [FakeBox()]

    def plot(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)


class FakeModel:
    names = {0: "person"}

    def __call__(self, *args, **kwargs):
        return [FakeResult()]


@pytest.fixture(autouse=True)
def setup_db_and_files(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_predictions.db")
    upload_dir = tmp_path / "original"
    predicted_dir = tmp_path / "predicted"

    upload_dir.mkdir()
    predicted_dir.mkdir()

    monkeypatch.setattr("app.DB_PATH", db_file)
    monkeypatch.setattr("app.UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr("app.PREDICTED_DIR", str(predicted_dir))
    monkeypatch.setattr("app.model", FakeModel())

    init_db()


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_image(tmp_path):
    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), color="white").save(image_path)
    return image_path


def test_predict_success(client, test_image):
    """
    Verify that uploading a valid image returns prediction data.
    """
    with open(test_image, "rb") as image_file:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_file, "image/jpeg")}
        )

    assert response.status_code == 200

    data = response.json()

    assert "prediction_uid" in data
    assert data["detection_count"] == 1
    assert data["labels"] == ["person"]
    assert "time_took" in data


def test_predict_rejects_non_image_file(client, tmp_path):
    """
    Verify that uploading a non-image file returns HTTP 400.
    """
    text_file = tmp_path / "test.txt"
    text_file.write_text("not an image")

    with open(text_file, "rb") as file:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", file, "text/plain")}
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only image files are supported"
    }