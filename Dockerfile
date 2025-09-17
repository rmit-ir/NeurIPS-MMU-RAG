# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Create a non-root user with home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser
# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Copy uv configuration files
COPY ./pyproject.toml ./uv.lock ./

# Install dependencies in a virtual environment
RUN mkdir ./src; uv sync --frozen --no-cache; rm -rf ./src

# Copy source code
COPY ./src ./src

# Run again with source code
RUN uv sync --frozen --no-cache

# Causing cache miss
# # Change ownership to non-root user
# RUN chown -R appuser:appuser /app

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
