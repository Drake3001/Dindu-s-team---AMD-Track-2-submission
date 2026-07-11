FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# System lib required by opencv-python-headless (libgthread from glib)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install locked dependencies only (do not build/install the project package)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# App code + baked pipeline config (NO credentials, videos, or test data)
COPY src/ ./src/
COPY config/ ./config/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

# main.py reads /input/tasks.json and writes /output/results.json
CMD ["python", "src/main.py"]
