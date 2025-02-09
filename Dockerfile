FROM public.ecr.aws/docker/library/python:3.10

# Install UV for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install linux dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

# Download and install ta-lib C library. Necessary to build python's from UV
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

# Create working directory /code
WORKDIR /app

# Copy the needed folders
COPY src/deployment /app/src/deployment
COPY src/support /app/src/support
COPY pyproject.toml /app
COPY uv.lock /app

# Create and activate a virtual environment using uv
RUN uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies
RUN uv sync

COPY src/data_etl_pipeline.py /app/src/data_etl_pipeline.py

ENV PYTHONPATH=/app

# On container run, execute train and predict scripts
CMD ["/bin/sh", "-c", "python -m src.deployment.train && python -m src.deployment.predict"]
