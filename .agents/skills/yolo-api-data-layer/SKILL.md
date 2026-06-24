````markdown
---
name: yolo-api-data-layer
description: >
  Use this skill for any task that touches the YOLO service database layer.
  Activate when the user asks to: refactor the API to use SQLAlchemy, add or
  modify endpoints that read or write predictions, add or modify database
  models or tables, make the database backend configurable (SQLite/Postgres),
  delete records, add columns to existing tables, or write tests for endpoints
  that depend on the database.

triggers:
  - "refactor the api to use sqlalchemy"
  - "add an endpoint GET /predictions/recent"
  - "add a UserFeedback table"
  - "write tests for the /predict endpoint"
  - "the database layer doesn't follow our architectural design"
  - "delete a prediction session"
  - "add a column processing_time_ms"
  - "make the database backend configurable"
  - "add a table"
  - "add a column"
  - "add an endpoint"
  - "database layer"
  - "sqlalchemy"
---

# YOLO API Data Layer Skill

## Service location

All work happens inside **`services/yolo/`**. Never modify files outside this directory.

## Architecture

The data layer is split across three files:

| File                      | Role                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `services/yolo/models.py` | SQLAlchemy ORM models only — no business logic             |
| `services/yolo/db.py`     | Engine creation, `SessionLocal`, and `get_db()` dependency |
| `services/yolo/app.py`    | FastAPI endpoints — imports from `models.py` and `db.py`   |

## Mandatory architecture requirements

The SQLAlchemy data layer MUST be separated from the FastAPI application code.

You MUST create and use these files:

- `services/yolo/models.py`
- `services/yolo/db.py`

Hard rules:

- Do NOT define SQLAlchemy ORM model classes inside `services/yolo/app.py`.
- Do NOT define `Base = declarative_base()` inside `services/yolo/app.py`.
- Do NOT define `create_engine()` inside `services/yolo/app.py`.
- Do NOT define `SessionLocal` inside `services/yolo/app.py`.
- Do NOT define `get_db()` inside `services/yolo/app.py`.
- Do NOT keep, rewrite, or call `init_db()`.
- Do NOT create model names like `DBPredictionSession` or `DBDetectionObject`.

The only valid ORM model class names are:

```python
PredictionSession
DetectionObject
```
````

In `app.py`, import them only with aliases:

```python
from models import PredictionSession as PredictionSessionORM
from models import DetectionObject as DetectionObjectORM
```

Never import ORM models without aliases.

A solution is incomplete if either `services/yolo/models.py` or `services/yolo/db.py` does not exist.

## Rules — follow these exactly

### 1. ORM only — no raw SQL

- Never use `import sqlite3`, `conn.execute(...)`, or raw SQL strings anywhere.
- All database reads and writes must go through SQLAlchemy ORM queries.

### 2. Models go in `models.py`

Define all ORM models in `services/yolo/models.py`. Use `declarative_base()` from `sqlalchemy.orm`:

```python
from sqlalchemy import Column, String, DateTime, Integer, Float
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class PredictionSession(Base):
    __tablename__ = "prediction_sessions"
    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_image = Column(String)
    predicted_image = Column(String)

class DetectionObject(Base):
    __tablename__ = "detection_objects"
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(String)
    label = Column(String)
    score = Column(Float)
    box = Column(String)
```

When adding a new table, add a new class to this file. When adding a column, add a new `Column(...)` to the relevant class.

### 3. Database connection goes in `db.py`

`services/yolo/db.py` must support SQLite (development) and Postgres (production) via environment variables:

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "pass")

if DB_BACKEND == "postgres":
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost/db"
else:
    DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 4. Endpoints use `Depends(get_db)`

Every endpoint that touches the database must declare a `db` parameter using FastAPI dependency injection:

```python
from fastapi import Depends
from sqlalchemy.orm import Session
from db import get_db

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
    session = db.query(PredictionSessionORM).filter_by(uid=uid).first()
    ...
```

Never call `get_db()` manually inside an endpoint.

Never open a `SessionLocal()` directly inside an endpoint.

### 5. Naming conflict — always alias ORM imports in `app.py`

`app.py` already contains a Pydantic model named `DetectionObject`.

The SQLAlchemy model in `models.py` has the same class name.

Always alias ORM imports to avoid naming collisions:

```python
from models import PredictionSession as PredictionSessionORM
from models import DetectionObject as DetectionObjectORM
```

Use:

- `PredictionSessionORM` for SQLAlchemy database operations
- `DetectionObjectORM` for SQLAlchemy database operations

Keep existing Pydantic model names unchanged.

Never rename the Pydantic `DetectionObject` model.

Never import ORM models without aliases.

### 6. Table creation replaces `init_db()`

Delete `init_db()` completely.

Do not redefine it.

Do not rewrite it.

Do not call it from tests.

Do not keep it for backward compatibility.

Tables are created automatically at startup.

Place this call at module level in `app.py`, after importing `Base` and `engine`:

```python
from db import get_db, engine
from models import Base

Base.metadata.create_all(bind=engine)
```

`init_db()` must not exist anywhere in the final solution.

### 7. Preserve all existing endpoints

All existing endpoint paths, HTTP methods, status codes, and response JSON shapes must remain exactly the same after any change.

Do not rename keys.

Do not change types.

Do not add fields.

Do not remove fields.

## Writing tests

### Test database setup

Use a temporary SQLite file per test.

Override the `get_db` dependency so the app uses the test session.

Never patch `app.DB_PATH`.

```python
# tests/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from models import Base
from db import get_db
from app import app

@pytest.fixture
def db_session(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def client(db_session):
    def override():
        yield db_session

    app.dependency_overrides[get_db] = override

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
```

### Inserting test data

Use ORM model instances and `db_session` directly.

```python
from models import PredictionSession as PredictionSessionORM
from models import DetectionObject as DetectionObjectORM

@pytest.fixture(autouse=True)
def seed_db(db_session):
    session = PredictionSessionORM(
        uid="abc-123",
        original_image="o.jpg",
        predicted_image="p.jpg"
    )

    obj = DetectionObjectORM(
        prediction_uid="abc-123",
        label="person",
        score=0.91,
        box="[10,20,100,200]"
    )

    db_session.add_all([session, obj])
    db_session.commit()
```

Never use `sqlite3.connect()` in tests.

### Mocking the YOLO model

For endpoints that call the YOLO model (e.g. `POST /predict`), patch `app.model` with a fake implementation.

## Verification — required before completion

Before declaring the task done, verify the required files exist:

```bash
test -f services/yolo/models.py
test -f services/yolo/db.py
```

Verify invalid patterns do not exist:

```bash
! grep -R "import sqlite3" services/yolo
! grep -R "from contextlib import closing" services/yolo
! grep -R "class DBPredictionSession" services/yolo
! grep -R "class DBDetectionObject" services/yolo
! grep -R "create_engine" services/yolo/app.py
! grep -R "SessionLocal" services/yolo/app.py
! grep -R "def get_db" services/yolo/app.py
! grep -R "def init_db" services/yolo
```

Then run:

```bash
cd services/yolo
pytest tests/ -v
```

All tests must pass.

If any required file is missing, any invalid pattern exists, or any test fails, the task is not complete.

```

```
