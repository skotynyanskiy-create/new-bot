FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    aiohttp>=3.13 orjson>=3.11 pydantic>=2.12 pydantic-settings>=2.13 \
    python-dotenv>=1.2 pyyaml>=6.0 websockets>=16.0 prometheus-client>=0.20

# App code
COPY . .

# Data directory
RUN mkdir -p data

# Health check
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default: paper trading
ENV PYTHONUNBUFFERED=1
CMD ["python", "src/main.py", "--capital", "500", "--log-level", "INFO"]
