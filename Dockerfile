FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY src/ src/

COPY main.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

RUN mkdir -p /logs

ENTRYPOINT ["./entrypoint.sh"]
