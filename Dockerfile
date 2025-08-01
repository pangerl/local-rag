# ---- Builder Stage ----
# This stage installs dependencies and generates the production requirements file.
FROM python:3.13-slim AS builder

# Install uv, a fast Python package installer and resolver.
RUN pip install uv

# Set the working directory.
WORKDIR /app

# Copy the project definition file.
COPY pyproject.toml .

# Generate the production requirements file, excluding development dependencies.
# This command compiles the dependencies specified in pyproject.toml into a requirements.txt format.
RUN uv pip compile pyproject.toml -o requirements.prod.txt

# ---- Final Stage ----
# This stage builds the final, lean production image.
FROM python:3.13-slim

# Set environment variables for Python.
# PYTHONDONTWRITEBYTECODE=1: Prevents Python from writing .pyc files.
# PYTHONUNBUFFERED=1: Ensures that Python output is sent straight to the terminal.
# PIP_NO_CACHE_DIR=1: Disables the pip cache.
# PIP_DISABLE_PIP_VERSION_CHECK=1: Disables pip version checks.
# PIP_ROOT_USER_ACTION=ignore: Ignores warnings about running pip as root.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Set the working directory.
WORKDIR /app

# Copy the generated production requirements file from the builder stage.
COPY --from=builder /app/requirements.prod.txt .

# Install system dependencies required for the application.
# - libmagic1: For file type detection.
# - libgl1-mesa-glx: For rendering in some PDF libraries.
# - tesseract-ocr: For Optical Character Recognition (OCR) in documents.
# - poppler-utils: For PDF processing and text extraction.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmagic1 \
        libgl1-mesa-glx \
        tesseract-ocr \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# 先强制安装 torch 的 CPU 版本，避免拉取 CUDA 相关依赖。
# RUN pip install --prefer-binary torch==2.7.0 --extra-index-url https://download.pytorch.org/whl/cpu
# 再安装其他依赖。
RUN pip install --prefer-binary -r requirements.prod.txt

# Copy the application code into the final image.
COPY . .

# Create a non-root user and change ownership of the app directory for security.
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to the non-root user.
USER appuser

# Expose the port the application will run on.
EXPOSE 8000

# Define the entrypoint and default command for the container.
# This enhances maintainability by separating the command from its arguments.
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
