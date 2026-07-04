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
    conf = [0.85]
    xyxy = [type("FakeXYXY", (), {"tolist": lambda self: [5, 10, 50, 80]})()]


class FakeResult:
    boxes = [FakeBox()]

    def plot(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)


class FakeModel:
    names = {0: "car"}

    def __call__(self, *args, **kwargs):
        return [FakeResult()]


def _write_jpeg_to_fileobj(bucket, key, fileobj):
    Image.new("RGB", (100, 100), color="blue").save(fileobj, format="JPEG")


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


def test_predict_includes_processing_time():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"image_s3_key": "originals/timing-test.jpg"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "processing_time_s" in data
    assert isinstance(data["processing_time_s"], (int, float))
    assert data["processing_time_s"] >= 0
