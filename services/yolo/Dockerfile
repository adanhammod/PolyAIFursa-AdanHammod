# ---------- Builder ----------
FROM python:3.12-slim-trixie AS builder

WORKDIR /app

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY torch-requirements.txt .
COPY requirements.txt .

RUN pip install --no-cache-dir -r torch-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# ---------- Runtime ----------
FROM python:3.12-slim-trixie

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]