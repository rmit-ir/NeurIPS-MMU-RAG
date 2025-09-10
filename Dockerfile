# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Create a non-root user with home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy uv configuration files
COPY ./pyproject.toml ./uv.lock ./

# Install dependencies in a virtual environment
RUN mkdir ./src; uv sync --frozen --no-cache; rm -rf ./src

# Copy source code
COPY ./src ./src

# Run again with source code
RUN uv sync --frozen --no-cache

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV UV_SYSTEM_PYTHON=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')" || exit 1

# Run the application
CMD ["uv", "run", "--frozen", "fastapi", "run", "src/apis/combined_app.py", "--host", "0.0.0.0", "--port", "8000"]
