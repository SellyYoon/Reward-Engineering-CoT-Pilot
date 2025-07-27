#!/bin/bash

# Usage: ./backup_docker_logs.sh [CONTAINER_NAME] [BACKUP_PATH]

# --- Argument Validation ---
# Check if exactly two arguments are provided.
if [ "$#" -ne 2 ]; then
    echo "Error: Two arguments are required (Container Name, Backup Path)"
    echo "Usage: $0 [CONTAINER_NAME] [BACKUP_PATH]"
    exit 1
fi

# --- Variable Initialization ---
CONTAINER_NAME=$1
BACKUP_DIR=$2
LOG_FILENAME="${CONTAINER_NAME}_docker.log"
DESTINATION_PATH="${BACKUP_DIR}/${LOG_FILENAME}"

echo "Starting Docker log backup for container: ${CONTAINER_NAME}"

# --- Main Logic ---
# 1. Get the full container ID from the container name.
CONTAINER_ID=$(docker ps -qf "name=${CONTAINER_NAME}")

if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Container '${CONTAINER_NAME}' not found."
    exit 1
fi

# 2. Define the default Docker log file path for Linux systems.
DOCKER_LOG_PATH="/var/lib/docker/containers/${CONTAINER_ID}/${CONTAINER_ID}-json.log"

# 3. Copy the log file.
#    Linux server
if [ -f "$DOCKER_LOG_PATH" ]; then
    cp "$DOCKER_LOG_PATH" "$DESTINATION_PATH"
    echo "Successfully backed up Docker log file to: ${DESTINATION_PATH}"
else
    echo "Log file not found at default path. Using 'docker logs' command as a fallback."
    docker logs "$CONTAINER_NAME" > "$DESTINATION_PATH" 2>&1
    echo "Successfully backed up Docker stream log to: ${DESTINATION_PATH}"
fi

exit 0