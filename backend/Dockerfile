FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for database connectivity
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for Docker
COPY requirements-docker.txt ./

# Install only essential dependencies (no heavy ML/AI packages)
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project files
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Copy entrypoint script
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]

