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

# Install the production dependencies.
# --prefer-binary: Prefer binary packages over source packages.
# -r requirements.prod.txt: Install packages from the specified file.
RUN pip install --prefer-binary -r requirements.prod.txt

# Copy the application code into the final image.
COPY . .

# Expose the port the application will run on.
EXPOSE 8000

# Define the entrypoint and default command for the container.
# This enhances maintainability by separating the command from its arguments.
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
