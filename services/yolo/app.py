<<<<<<< HEAD
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
from PIL import Image
from contextlib import closing
from pydantic import BaseModel, field_validator
from datetime import datetime, timezone
from typing import Optional
import json

import sys
import signal
import sqlite3
import logging
import os
import uuid
import time
import boto3

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class DetectionObject(BaseModel):
    id: int
    label: str
    score: float
    box: list[float]

    @field_validator("box", mode="before")
    @classmethod
    def parse_box(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class PredictRequest(BaseModel):
    image_s3_key: str


class PredictResponse(BaseModel):
    uid: str
    timestamp: datetime
    original_image: str
    predicted_image: str
    annotated_image_s3_key: Optional[str] = None
    detection_objects: list[DetectionObject]
    processing_time_s: float


# Disable GPU usage
import torch

torch.cuda.is_available = lambda: False

app = FastAPI()


is_shutting_down = False


def handle_sigterm(signum, frame):
    global is_shutting_down
    is_shutting_down = True
    logging.info("Received SIGTERM. Shutting down gracefully...")
    # Perform cleanup: close DB connections, finish pending work, etc.
    logging.info("Cleanup done. Exiting.")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)

# Confidence threshold for object detection (0.0 - 1.0).
# Detections below this score are discarded.
# Override with: export CONFIDENCE_THRESHOLD=0.7
_raw_threshold = os.environ.get("CONFIDENCE_THRESHOLD")
if _raw_threshold is not None:
    CONFIDENCE_THRESHOLD = float(_raw_threshold)
    logging.info(
        f"CONFIDENCE_THRESHOLD set to {CONFIDENCE_THRESHOLD} (from environment)"
    )
else:
    CONFIDENCE_THRESHOLD = 0.5
    logging.info(f"CONFIDENCE_THRESHOLD not set, using default: {CONFIDENCE_THRESHOLD}")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
s3_client = boto3.client("s3", region_name=AWS_REGION)

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")


# Initialize SQLite
def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)

        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)

        # Create index for faster queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)"
        )

        conn.commit()


def save_prediction_session(uid, original_image, predicted_image):
    """
    Save prediction session to database
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """,
            (uid, original_image, predicted_image),
        )

        conn.commit()


def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """,
            (prediction_uid, label, score, str(box)),
        )

        conn.commit()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()

    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ".jpg")
    predicted_path = os.path.join(PREDICTED_DIR, uid + ".jpg")

    with open(original_path, "wb") as f:
        s3_client.download_fileobj(AWS_S3_BUCKET, request.image_s3_key, f)

    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    annotated_key = f"predicted/{uid}.jpg"
    with open(predicted_path, "rb") as f:
        s3_client.upload_fileobj(f, AWS_S3_BUCKET, annotated_key)

    save_prediction_session(uid, original_path, predicted_path)

    detection_objects = []
    for idx, box in enumerate(results[0].boxes):
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
        detection_objects.append(DetectionObject(id=idx, label=label, score=score, box=bbox))

    return PredictResponse(
        uid=uid,
        timestamp=datetime.now(timezone.utc),
        original_image=original_path,
        predicted_image=predicted_path,
        annotated_image_s3_key=annotated_key,
        detection_objects=detection_objects,
        processing_time_s=round(time.time() - start_time, 2),
    )


# prediction endpoint
@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    """
    Get prediction session by uid with all detected objects
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute(
            "SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")

        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", (uid,)
        ).fetchall()

        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"],
                }
                for obj in objects
            ],
        }


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str):
    """
    Return the annotated (bounding-box) image for a prediction
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)
        ).fetchone()
    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(row[0])


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    """
    Return all prediction sessions containing at least one detected object with the given label
    """
    if not label or label.strip() == "":
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row

        # Find all prediction UIDs that have at least one object with this label
        sessions_with_label = conn.execute(
            """
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            INNER JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ?
        """,
            (label,),
        ).fetchall()

        results = []
        for session in sessions_with_label:
            # Get all detection objects with the matching label for this session
            objects = conn.execute(
                """
                SELECT id, label, score, box
                FROM detection_objects
                WHERE prediction_uid = ? AND label = ?
            """,
                (session["uid"], label),
            ).fetchall()

            results.append(
                {
                    "uid": session["uid"],
                    "timestamp": session["timestamp"],
                    "detection_objects": [
                        {
                            "id": obj["id"],
                            "label": obj["label"],
                            "score": obj["score"],
                            "box": obj["box"],
                        }
                        for obj in objects
                    ],
                }
            )

        return results


@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):
    """
    Return all detection objects with score >= min_score.
    min_score must be between 0.0 and 1.0.
    """

    if not 0.0 <= min_score <= 1.0:
        raise HTTPException(
            status_code=400, detail="min_score must be between 0.0 and 1.0"
        )

    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT id, prediction_uid, label, score, box
            FROM detection_objects
            WHERE score >= ?
        """,
            (min_score,),
        ).fetchall()

        return [
            {
                "id": row["id"],
                "prediction_uid": row["prediction_uid"],
                "label": row["label"],
                "score": row["score"],
                "box": row["box"],
            }
            for row in rows
        ]


@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}


@app.get("/health2")
def health2():
    """
    Health check endpoint
    """
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if is_shutting_down:
        raise HTTPException(status_code=503, detail="Service is shutting down")
    return {"status": "ready"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    init_db()

    uvicorn.run(app, host="0.0.0.0", port=8080)
=======
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
from PIL import Image
from contextlib import closing
from pydantic import BaseModel, field_validator
from datetime import datetime, timezone
from typing import Optional
import json

import sys
import signal
import sqlite3
import logging
import os
import uuid
import shutil
import time
import boto3

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class DetectionObject(BaseModel):
    id: int
    label: str
    score: float
    box: list[float]

    @field_validator("box", mode="before")
    @classmethod
    def parse_box(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class PredictRequest(BaseModel):
    image_s3_key: str


class PredictResponse(BaseModel):
    uid: str
    timestamp: datetime
    original_image: str
    predicted_image: str
    annotated_image_s3_key: Optional[str] = None
    detection_objects: list[DetectionObject]
    processing_time_s: float


# Disable GPU usage
import torch

torch.cuda.is_available = lambda: False

app = FastAPI()


is_shutting_down = False


def handle_sigterm(signum, frame):
    global is_shutting_down
    is_shutting_down = True
    logging.info("Received SIGTERM. Shutting down gracefully...")
    # Perform cleanup: close DB connections, finish pending work, etc.
    logging.info("Cleanup done. Exiting.")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)

# Confidence threshold for object detection (0.0 - 1.0).
# Detections below this score are discarded.
# Override with: export CONFIDENCE_THRESHOLD=0.7
_raw_threshold = os.environ.get("CONFIDENCE_THRESHOLD")
if _raw_threshold is not None:
    CONFIDENCE_THRESHOLD = float(_raw_threshold)
    logging.info(
        f"CONFIDENCE_THRESHOLD set to {CONFIDENCE_THRESHOLD} (from environment)"
    )
else:
    CONFIDENCE_THRESHOLD = 0.5
    logging.info(f"CONFIDENCE_THRESHOLD not set, using default: {CONFIDENCE_THRESHOLD}")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
s3_client = boto3.client("s3", region_name=AWS_REGION)

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")


# Initialize SQLite
def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)

        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)

        # Create index for faster queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)"
        )

        conn.commit()


def save_prediction_session(uid, original_image, predicted_image):
    """
    Save prediction session to database
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """,
            (uid, original_image, predicted_image),
        )

        conn.commit()


def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """,
            (prediction_uid, label, score, str(box)),
        )

        conn.commit()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()

    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ".jpg")
    predicted_path = os.path.join(PREDICTED_DIR, uid + ".jpg")

    with open(original_path, "wb") as f:
        s3_client.download_fileobj(AWS_S3_BUCKET, request.image_s3_key, f)

    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    annotated_key = f"predicted/{uid}.jpg"
    with open(predicted_path, "rb") as f:
        s3_client.upload_fileobj(f, AWS_S3_BUCKET, annotated_key)

    save_prediction_session(uid, original_path, predicted_path)

    detection_objects = []
    for idx, box in enumerate(results[0].boxes):
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
        detection_objects.append(
            DetectionObject(id=idx, label=label, score=score, box=bbox)
        )

    return PredictResponse(
        uid=uid,
        timestamp=datetime.now(timezone.utc),
        original_image=original_path,
        predicted_image=predicted_path,
        annotated_image_s3_key=annotated_key,
        detection_objects=detection_objects,
        processing_time_s=round(time.time() - start_time, 2),
    )


# prediction endpoint
@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    """
    Get prediction session by uid with all detected objects
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute(
            "SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")

        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", (uid,)
        ).fetchall()

        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"],
                }
                for obj in objects
            ],
        }


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str):
    """
    Return the annotated (bounding-box) image for a prediction
    """
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)
        ).fetchone()
    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(row[0])


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    """
    Return all prediction sessions containing at least one detected object with the given label
    """
    if not label or label.strip() == "":
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row

        # Find all prediction UIDs that have at least one object with this label
        sessions_with_label = conn.execute(
            """
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            INNER JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ?
        """,
            (label,),
        ).fetchall()

        results = []
        for session in sessions_with_label:
            # Get all detection objects with the matching label for this session
            objects = conn.execute(
                """
                SELECT id, label, score, box
                FROM detection_objects
                WHERE prediction_uid = ? AND label = ?
            """,
                (session["uid"], label),
            ).fetchall()

            results.append(
                {
                    "uid": session["uid"],
                    "timestamp": session["timestamp"],
                    "detection_objects": [
                        {
                            "id": obj["id"],
                            "label": obj["label"],
                            "score": obj["score"],
                            "box": obj["box"],
                        }
                        for obj in objects
                    ],
                }
            )

        return results


@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):
    """
    Return all detection objects with score >= min_score.
    min_score must be between 0.0 and 1.0.
    """

    if not 0.0 <= min_score <= 1.0:
        raise HTTPException(
            status_code=400, detail="min_score must be between 0.0 and 1.0"
        )

    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT id, prediction_uid, label, score, box
            FROM detection_objects
            WHERE score >= ?
        """,
            (min_score,),
        ).fetchall()

        return [
            {
                "id": row["id"],
                "prediction_uid": row["prediction_uid"],
                "label": row["label"],
                "score": row["score"],
                "box": row["box"],
            }
            for row in rows
        ]


@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}


@app.get("/health2")
def health2():
    """
    Health check endpoint
    """
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if is_shutting_down:
        raise HTTPException(status_code=503, detail="Service is shutting down")
    return {"status": "ready"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    init_db()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        loop="asyncio",
    )
>>>>>>> feature/t004-img-proc-mcp
