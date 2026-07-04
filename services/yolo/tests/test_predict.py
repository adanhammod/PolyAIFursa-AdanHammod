import io
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")
os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")

from app import app, init_db  # noqa: E402


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


def _write_jpeg_to_fileobj(bucket, key, fileobj):
    Image.new("RGB", (100, 100), color="white").save(fileobj, format="JPEG")


@pytest.fixture(autouse=True)
def mock_s3(monkeypatch):
    fake = MagicMock()
    fake.download_fileobj.side_effect = _write_jpeg_to_fileobj
    monkeypatch.setattr("app.s3_client", fake)
    return fake


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


def test_predict_success(client, mock_s3):
    response = client.post("/predict", json={"image_s3_key": "originals/test.jpg"})

    assert response.status_code == 200

    data = response.json()

    assert "uid" in data
    assert "timestamp" in data
    assert "original_image" in data
    assert "predicted_image" in data
    assert "detection_objects" in data
    assert len(data["detection_objects"]) == 1
    assert data["detection_objects"][0]["label"] == "person"
    assert "processing_time_s" in data
    assert "annotated_image_s3_key" in data
    assert data["annotated_image_s3_key"].startswith("predicted/")
    assert mock_s3.upload_fileobj.call_count == 1
