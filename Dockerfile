# syntax=docker/dockerfile:1
# Model-serving image for the heart-disease risk API.
# Build context = repository root. The trained artifact (artifacts/model/) must exist:
#   python scripts/download_data.py && python -m heartrisk.data && python -m heartrisk.train
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/artifacts/model \
    LOG_LEVEL=INFO \
    PORT=8000

WORKDIR /app

# 1) dependencies first -> cached layer unless requirements change
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2) application code and the packaged model (preprocessing + classifier in one pipeline)
COPY src/ ./src/
COPY artifacts/model/ ./artifacts/model/

# 3) run as an unprivileged user. USER is numeric so Kubernetes (runAsNonRoot) can verify
#    the container is not root without resolving a user name from /etc/passwd.
RUN groupadd --gid 10001 appuser \
 && useradd --uid 10001 --gid 10001 --create-home appuser \
 && chown -R 10001:10001 /app
USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status==200 else 1)"

CMD ["sh", "-c", "uvicorn heartrisk.api.app:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
