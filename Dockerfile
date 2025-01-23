FROM python:3.10

# Installing uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create working directory /code
WORKDIR /app

COPY . /app

# Install dependencies
RUN uv sync --frozen --no-cache

# Run the application
CMD ["/app/.venv/bin/uvicorn",  "predict:app", "--port", "8000","--host","0.0.0.0"]