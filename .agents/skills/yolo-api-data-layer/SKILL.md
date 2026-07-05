---
name: yolo-api-data-layer
description: >
  Activate for any task that touches the YOLO service database layer:
  adding or modifying SQLAlchemy models, creating or refactoring database
  endpoints, making the backend configurable (SQLite/PostgreSQL), writing
  or refactoring database-related tests, or performing a full data-layer
  architecture refactor.

triggers:
  - "refactor the api to use sqlalchemy"
  - "add a UserFeedback table"
  - "add a new database column"
  - "create a new endpoint that queries the database"
  - "delete a prediction session"
  - "write tests for a database endpoint"
  - "migrate sqlite code to postgres"
  - "fix the api data layer architecture"
  - "add a column"
  - "add a table"
  - "add an endpoint"
  - "database layer"
  - "sqlalchemy"
  - "make the database backend configurable"
  - "write tests"
  - "add tests"
  - "test the"
  - "fix the architecture"
  - "architectural design"
  - "delete a prediction"
  - "database backend configurable"
  - "use postgres"
  - "remove a prediction"
  - "remove prediction session"
  - "delete session"
  - "delete by uid"
  - "delete all detection objects"
---

## STOP — Read This Before Writing Any Code

## MANDATORY FIRST ACTION — Architecture Check

Before reading any existing code, run this check:

    test -f services/yolo/models.py && test -f services/yolo/db.py && echo "OK" || echo "REFACTOR REQUIRED"

If the output is "REFACTOR REQUIRED":

- Stop immediately
- Do NOT read app.py for patterns
- Do NOT match existing code style
- The existing code is LEGACY and must not be imitated
- Perform the full SQLAlchemy refactor first (see Task: Full data-layer refactor in this skill)
- Only after both files exist on disk, proceed with the requested feature

Step 1: Create services/yolo/models.py
Step 2: Create services/yolo/db.py
Step 3: Only then edit services/yolo/app.py

If you edit app.py before both files exist on disk, you have failed.
If models.py does not exist when you finish, you have failed.
If db.py does not exist when you finish, you have failed.
Modifying app.py is not a substitute for creating these files.

---

## BEFORE ADDING ANY ENDPOINT — Check Architecture First

Before writing any new endpoint that touches the database:

1. Check if services/yolo/models.py exists
2. Check if services/yolo/db.py exists

If either file is missing:

- Do NOT add the endpoint using raw SQL or sqlite3
- Do NOT match the existing code style if it uses sqlite3
- STOP and perform the full data layer refactor first (create models.py,
  create db.py, rewrite app.py to use ORM)
- Only then add the new endpoint using Depends(get_db) and ORM queries

A new endpoint added on top of raw sqlite3 code is invalid even if
its tests pass.

---

# YOLO API Data Layer Skill

## DO THIS FIRST — Mandatory Three-File Split

Before changing any endpoint logic, create these files:

1. services/yolo/models.py
2. services/yolo/db.py

Then update:

3. services/yolo/app.py

Never define SQLAlchemy models, create_engine, SessionLocal, or get_db inside app.py.
If any of those exist in app.py after the refactor, the task is failed.

## Purpose

This skill governs all database-related work in the YOLO FastAPI service located at
`services/yolo/`. It enforces a strict three-file data-layer architecture using
SQLAlchemy ORM, FastAPI dependency injection, and repository-style data access
patterns. Its goal is to ensure every data-layer change is:

- Isolated from application business logic
- Portable between SQLite (development) and PostgreSQL (production)
- Tested with proper dependency injection — no raw SQL, no sqlite3 module
- Backward-compatible: existing API consumers must never be broken

All work described in this skill happens **exclusively inside `services/yolo/`**.
Never modify files outside this directory.

---

## When To Use

Activate this skill whenever the task involves any of the following:

- Refactoring the API to adopt SQLAlchemy ORM
- Adding, renaming, or removing a database table
- Adding, renaming, or removing a column on an existing table
- Creating a new FastAPI endpoint that reads from or writes to the database
- Deleting records from the database
- Writing or refactoring tests for database-backed endpoints
- Making the database backend configurable (SQLite ↔ PostgreSQL)
- Diagnosing or correcting architecture violations in the data layer

Do **not** activate this skill for changes that are purely about request validation,
response serialisation, YOLO model inference, or file upload handling, unless those
changes also touch the database layer.

---

## Architecture Rules

The data layer is split across exactly three files:

| File                      | Responsibility                                            |
| ------------------------- | --------------------------------------------------------- |
| `services/yolo/models.py` | SQLAlchemy ORM model definitions only — no business logic |
| `services/yolo/db.py`     | Engine, SessionLocal, and get_db() — no models            |
| `services/yolo/app.py`    | FastAPI routes — imports from models.py and db.py         |

### Absolute prohibitions in `app.py`

- Do NOT define `Base = declarative_base()` in `app.py`
- Do NOT define any SQLAlchemy model class in `app.py`
- Do NOT call `create_engine()` in `app.py`
- Do NOT define or assign `SessionLocal` in `app.py`
- Do NOT define `get_db()` in `app.py`
- Do NOT define or call `init_db()` anywhere — delete it permanently
- Do NOT use `import sqlite3` anywhere in the service
- Do NOT use raw SQL strings (`SELECT`, `INSERT`, `UPDATE`, `DELETE`, `CREATE TABLE`, `ALTER TABLE`)
- Do NOT name ORM models `DBPredictionSession`, `DBDetectionObject`, or any `DB`-prefixed variant

### Mandatory ORM model names

The only valid class names for the two core ORM models are:

```python
PredictionSession
DetectionObject
```

Because `app.py` already has a Pydantic model named `DetectionObject`, all ORM imports
into `app.py` **must** use aliases:

```python
from models import PredictionSession as PredictionSessionORM
from models import DetectionObject as DetectionObjectORM
```

Never import ORM models into `app.py` without the `ORM` alias suffix.

### Table creation

Tables are created at application startup via a module-level call in `app.py`:

```python
from db import get_db, engine
from models import Base

Base.metadata.create_all(bind=engine)
```

`init_db()` must not exist anywhere in the repository.

---

## Required File Structure

### `services/yolo/models.py`

```python
from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey, Index
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
    prediction_uid = Column(String, ForeignKey("prediction_sessions.uid"))
    label = Column(String)
    score = Column(Float)
    box = Column(String)

    __table_args__ = (
        Index("idx_prediction_uid", "prediction_uid"),
        Index("idx_label", "label"),
        Index("idx_score", "score"),
    )
```

When adding a new table, add a new class to this file.
When adding a column, add a `Column(...)` to the relevant class.
Never touch `db.py` or `app.py` for schema changes.

---

### `services/yolo/db.py`

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

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

Environment variables:

- `DB_BACKEND` — `"sqlite"` (default) or `"postgres"`
- `DB_USER` — PostgreSQL username
- `DB_PASSWORD` — PostgreSQL password

No code change is required when switching backends.

---

### `services/yolo/app.py` (imports section — required shape)

```python
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from db import get_db, engine
from models import Base
from models import PredictionSession as PredictionSessionORM
from models import DetectionObject as DetectionObjectORM

Base.metadata.create_all(bind=engine)
```

---

## SQLAlchemy Standards

### Allowed ORM operations

```python
# Read
db.query(PredictionSessionORM).filter(...).first()
db.query(PredictionSessionORM).filter(...).all()
db.query(PredictionSessionORM).order_by(...).limit(n).all()

# Write
db.add(PredictionSessionORM(...))
db.commit()

# Delete
obj = db.query(PredictionSessionORM).filter_by(uid=uid).first()
db.delete(obj)
db.commit()
```

### Forbidden patterns

```python
# Never — raw SQL
db.execute("SELECT * FROM prediction_sessions")
conn.execute("INSERT INTO ...")
"SELECT uid FROM prediction_sessions WHERE ..."

# Never — sqlite3 module
import sqlite3
sqlite3.connect("predictions.db")

# Never — manual session management in routes
db = SessionLocal()   # inside an endpoint function
```

### Adding a new table

1. Add a new class to `models.py` inheriting from `Base`
2. Define `__tablename__` and all columns
3. Add a foreign key to `prediction_sessions.uid` if the table is linked to predictions
4. `Base.metadata.create_all(bind=engine)` in `app.py` will create the table on next startup
5. Add `from models import NewModel as NewModelORM` in `app.py` only if the endpoint needs it

### Adding a new column

1. Add `Column(...)` to the relevant class in `models.py`
2. Update the endpoint that populates it in `app.py`
3. For SQLite development: drop and recreate the DB file (or use Alembic for migrations)
4. Never write `ALTER TABLE` statements

---

## FastAPI Standards

### Dependency injection — mandatory pattern

Every endpoint that touches the database must include:

```python
db: Session = Depends(get_db)
```

Example:

```python
@app.get("/predictions/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
    session = db.query(PredictionSessionORM).filter_by(uid=uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")
    ...
```

### Prohibited patterns in routes

```python
# Never — bypasses DI
db = SessionLocal()

# Never — raw module
sqlite3.connect("predictions.db")

# Never — calling the generator manually
db = next(get_db())
```

### Endpoint compatibility

All existing endpoint paths, HTTP methods, status codes, and response JSON shapes must remain
**exactly the same** after any change. Protected endpoints:

- `POST /predict`
- `GET /prediction/{uid}`
- `GET /prediction/{uid}/image`
- `GET /predictions/label/{label}`
- `GET /predictions/score/{min_score}`
- `GET /health`
- `GET /health2`
- `GET /ready`

Do not rename keys, change types, add fields, or remove fields from any existing response.

---

## Testing Standards

### Test setup — conftest.py (mandatory)

Create `services/yolo/tests/conftest.py`:

```python
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

### Seeding test data — ORM only

```python
from models import PredictionSession as PredictionSessionORM
from models import DetectionObject as DetectionObjectORM


@pytest.fixture(autouse=True)
def seed_db(db_session):
    session = PredictionSessionORM(
        uid="abc-123",
        original_image="original.jpg",
        predicted_image="predicted.jpg",
    )
    obj = DetectionObjectORM(
        prediction_uid="abc-123",
        label="person",
        score=0.91,
        box="[10, 20, 100, 200]",
    )
    db_session.add_all([session, obj])
    db_session.commit()
```

### Mocking the YOLO model (for /predict tests)

```python
import numpy as np


class FakeBox:
    cls = [type("V", (), {"item": lambda self: 0})()]
    conf = [0.91]
    xyxy = [type("V", (), {"tolist": lambda self: [10, 20, 100, 200]})()]


class FakeResult:
    boxes = [FakeBox()]

    def plot(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)


class FakeModel:
    names = {0: "person"}

    def __call__(self, *args, **kwargs):
        return [FakeResult()]


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    monkeypatch.setattr("app.model", FakeModel())
```

### Absolute prohibitions in tests

- `import sqlite3` — forbidden
- `sqlite3.connect(...)` — forbidden
- `init_db()` — forbidden (it no longer exists)
- `app.DB_PATH` — forbidden (no such attribute)
- `monkeypatch.setattr("app.DATABASE_URL", ...)` — forbidden
- Raw SQL strings for seeding — forbidden

Existing tests that use any of these patterns must be rewritten to use `conftest.py` fixtures.

---

## Verification Checklist

Run these checks before declaring any data-layer task complete.

### File existence

```bash
test -f services/yolo/models.py
test -f services/yolo/db.py
test -f services/yolo/tests/conftest.py
```

### Architecture violations — none must match

```bash
! grep -R "import sqlite3"              services/yolo/
! grep -R "sqlite3.connect"            services/yolo/
! grep -R "class DBPredictionSession"  services/yolo/
! grep -R "class DBDetectionObject"    services/yolo/
! grep -R "def init_db"                services/yolo/
! grep -R "app\.DB_PATH"               services/yolo/
! grep -rE "\bSELECT\b"               services/yolo/app.py
! grep -rE "\bINSERT\b"               services/yolo/app.py
! grep -rE "\bUPDATE\b"               services/yolo/app.py
! grep -rE "\bDELETE\b"               services/yolo/app.py
! grep    "create_engine"              services/yolo/app.py
! grep    "SessionLocal"               services/yolo/app.py
! grep    "def get_db"                 services/yolo/app.py
```

### Required patterns — all must match

```bash
grep "class PredictionSession(Base)"  services/yolo/models.py
grep "class DetectionObject(Base)"    services/yolo/models.py
grep "def get_db"                     services/yolo/db.py
grep "SessionLocal"                   services/yolo/db.py
grep "create_engine"                  services/yolo/db.py
grep "DB_BACKEND"                     services/yolo/db.py
grep "Base.metadata.create_all"       services/yolo/app.py
grep "Depends(get_db)"                services/yolo/app.py
grep "PredictionSession as PredictionSessionORM"  services/yolo/app.py
grep "DetectionObject as DetectionObjectORM"      services/yolo/app.py
grep "dependency_overrides"           services/yolo/tests/conftest.py
```

### Tests

```bash
cd services/yolo && pytest tests/ -v
```

All tests must pass (exit code 0). The task is not complete until this is confirmed.

---

## Common Tasks

### Task: Full data-layer refactor

1. Create `services/yolo/models.py` with `Base`, `PredictionSession`, `DetectionObject`
2. Create `services/yolo/db.py` with `engine`, `SessionLocal`, `get_db()`, SQLite/PostgreSQL config
3. In `app.py`:
   - Remove all ORM model class definitions
   - Remove `Base = declarative_base()`
   - Remove `create_engine`, `SessionLocal`, `get_db`, `init_db`
   - Remove `import sqlite3` (if present)
   - Add imports: `from db import get_db, engine` and `from models import Base`
   - Add `from models import PredictionSession as PredictionSessionORM`
   - Add `from models import DetectionObject as DetectionObjectORM`
   - Add `Base.metadata.create_all(bind=engine)` at module level
   - Replace `DBPredictionSession` → `PredictionSessionORM` throughout
   - Replace `DBDetectionObject` → `DetectionObjectORM` throughout
4. Create `services/yolo/tests/conftest.py` with `db_session` and `client` fixtures
5. Rewrite all test files to use `conftest.py` fixtures
6. Run verification checklist
7. Run `pytest tests/ -v`

### Task: Add a new table

1. Add a new class to `services/yolo/models.py` inheriting from `Base`
2. If the table references predictions, add `ForeignKey("prediction_sessions.uid")`
3. If new endpoints are needed, add them to `app.py` using `Depends(get_db)`
4. Import the new model with an `ORM` alias in `app.py` if used in routes
5. Write tests using `conftest.py` fixtures
6. Run `pytest tests/ -v`

### Task: Add a column to an existing table

1. Add `Column(...)` to the relevant class in `services/yolo/models.py`
2. Update the endpoint that populates it in `app.py`
3. For development: delete `predictions.db` so it is recreated with the new schema
4. Run `pytest tests/ -v`

### Task: Add a new database-backed endpoint

1. Add the route to `app.py` with `db: Session = Depends(get_db)`
2. Query using `db.query(ModelORM).filter(...).all()` — no raw SQL
3. Return JSON matching the established response shape for the resource
4. Write tests in a new or existing test file using `conftest.py` fixtures
5. Run `pytest tests/ -v`

### Task: Make the database backend configurable

1. Ensure `services/yolo/db.py` reads `DB_BACKEND`, `DB_USER`, `DB_PASSWORD`
2. Ensure `connect_args={"check_same_thread": False}` is only passed for SQLite
3. Ensure no database URL or credentials appear in `app.py`
4. Document environment variables in the service README
5. Run `pytest tests/ -v`

---

## Examples

### Example 1 — "refactor the api to use sqlalchemy"

The agent must:

- Create `models.py` and `db.py` as physical files
- Strip all ORM and DB connection code from `app.py`
- Add aliased imports and `Base.metadata.create_all(bind=engine)`
- Create `tests/conftest.py` and rewrite all test fixtures
- Confirm `pytest tests/ -v` exits with code 0

The agent must **not**:

- Leave any ORM class definition in `app.py`
- Keep `init_db()` anywhere
- Preserve old test patterns that call `init_db()` or patch `app.DB_PATH`

---

### Example 2 — "add a UserFeedback table to track user ratings per prediction"

The agent must:

- Add `UserFeedback(Base)` to `models.py` with at minimum:
  `id`, `prediction_uid` (FK), `rating` (Integer), `created_at` (DateTime)
- Not modify any existing endpoint
- Not write any `CREATE TABLE` SQL
- Let `Base.metadata.create_all(bind=engine)` handle table creation at startup
- Confirm `pytest tests/ -v` still passes

---

### Example 3 — "add an endpoint GET /predictions/recent"

The agent must:

- Add `@app.get("/predictions/recent")` to `app.py`
- Use `db: Session = Depends(get_db)` in the signature
- Query `PredictionSessionORM` ordered by `timestamp` descending, limited to 10
- Return a JSON list with at least `uid` and `timestamp` fields
- Add a test for the new endpoint using `conftest.py` fixtures
- Confirm existing endpoints are unchanged
- Confirm `pytest tests/ -v` exits with code 0

---

### Example 4 — "add a processing_time_ms column to the predictions table"

The agent must:

- Add `processing_time_ms = Column(Float, nullable=True)` to `PredictionSession` in `models.py`
- Update `POST /predict` in `app.py` to populate the column when saving
- Not change the `PredictResponse` schema (field is internal storage only unless asked)
- Not write `ALTER TABLE` SQL
- Confirm `pytest tests/ -v` exits with code 0

---

### Example 5 — "write tests for the /predict endpoint"

The agent must:

- Ensure `tests/conftest.py` exists with `db_session` and `client` fixtures
- Add `FakeModel`, `FakeBox`, `FakeResult` mock classes
- Wire the model mock via `monkeypatch.setattr("app.model", FakeModel())`
- Assert HTTP 200 for a valid image upload
- Assert HTTP 400 for a non-image file upload
- Not use `sqlite3.connect`, `init_db`, or `app.DB_PATH` anywhere
- Confirm `pytest tests/ -v` exits with code 0
