FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src/ src/
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

COPY main.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

RUN mkdir -p /logs

ENTRYPOINT ["./entrypoint.sh"]
