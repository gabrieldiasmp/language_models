#!/bin/bash
set -e

# 🔍 Automatically detect the absolute path of the repository root
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOLUME_PATH="$REPO_PATH/docker_volumes"

echo "#### Repository path detected: $REPO_PATH"
echo "#### Volume base directory: $VOLUME_PATH"

# Create all required volume directories if they don't exist
mkdir -p \
  "$VOLUME_PATH/postgres_data" \
  "$VOLUME_PATH/opensearch_data" \
  "$VOLUME_PATH/ollama_data" \
  "$VOLUME_PATH/airflow_logs" \
  "$VOLUME_PATH/langfuse_postgres_data" \
  "$VOLUME_PATH/clickhouse_data" \
  "$VOLUME_PATH/redis_data" \
  "$VOLUME_PATH/cloudbeaver_data"

echo "#### All volume directories created successfully."

# Export REPO_PATH so docker compose can use it
export REPO_PATH