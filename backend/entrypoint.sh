#!/bin/bash
set -e

echo "Waiting for database to be ready..."
# Use psql to check database connection
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; do
  echo "Database is unavailable - sleeping..."
  sleep 2
done

echo "✅ Database is ready!"

echo "Running Alembic migrations..."
alembic upgrade head

echo "✅ Migrations completed!"
echo "Starting FastAPI application..."
uvicorn portfoliopilot.main:app --host 0.0.0.0 --port 8000

