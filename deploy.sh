#!/bin/bash

# Exit script on any error
set -e

echo "==========================================="
echo "   Hoax Detection App Deployment Script    "
echo "==========================================="

echo "[1/3] Building the Docker image (this may take a few minutes)..."
docker build -t hoax-detection-app .

echo "[2/3] Checking for existing running container..."
if [ "$(docker ps -q -f name=hoax-app)" ]; then
    echo "Stopping existing container..."
    docker stop hoax-app
fi

if [ "$(docker ps -aq -f status=exited -f name=hoax-app)" ]; then
    echo "Removing existing container..."
    docker rm hoax-app
fi

echo "[3/3] Starting the Docker container..."
docker run -d -p 8505:8505 --name hoax-app hoax-detection-app

echo "==========================================="
echo "Deployment successful! 🎉"
echo "You can view the app at: http://localhost:8505"
echo "To view logs, run: docker logs -f hoax-app"
echo "To stop the app, run: docker stop hoax-app"
echo "==========================================="
