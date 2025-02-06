FROM public.ecr.aws/docker/library/python:3.10

# Installing uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

# Download, extract, and install ta-lib
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

COPY src/deployment /app
COPY pyproject.toml /app
COPY uv.lock /app


# Install dependencies
RUN uv sync

ENV PYTHONPATH=/app
