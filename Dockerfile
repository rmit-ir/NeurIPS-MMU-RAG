# Use cuda heavy duty image, devel is needed for flash attention compilation
FROM docker.io/ubuntu:24.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# pkg-config: needed for Rust compiler to find libssl-dev
RUN apt update \
    && apt install -y curl numactl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy uv configuration files
COPY ./pyproject.toml ./uv.lock ./

# Install dependencies in a virtual environment
RUN mkdir ./src; uv sync --frozen --no-cache --extra vllm --extra classifier; rm -rf ./src

# Copy source code
COPY ./src ./src

# Run again with source code
RUN uv sync --frozen --no-cache --extra vllm --extra classifier

# Copy ML models
COPY ./models ./models

# Expose port
EXPOSE 5025

# Set environment variables
ENV PYTHONPATH=/app/src
ENV UV_SYSTEM_PYTHON=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5025/health/')" || exit 1

# Run the application
CMD ["uv", "run", "--frozen", "fastapi", "run", "src/apis/mmu_rag_router.py", "--host", "0.0.0.0", "--port", "5025"]
