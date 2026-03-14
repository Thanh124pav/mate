#!/bin/bash

# Script to build and push MATE Docker image

# Configuration
IMAGE_NAME="mate"
DOCKER_USERNAME=""  # Replace with your Docker Hub username
TAG="latest"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--username)
      DOCKER_USERNAME="$2"
      shift 2
      ;;
    -t|--tag)
      TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if Docker username is provided
if [ -z "$DOCKER_USERNAME" ]; then
  echo "Error: Docker username is required"
  echo "Usage: $0 -u <docker-username> [-t <tag>]"
  echo "Example: $0 -u myusername -t v1.0"
  exit 1
fi

# Full image name
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"
echo "=================================="

# Build the Docker image
docker build -t ${FULL_IMAGE_NAME} .

if [ $? -ne 0 ]; then
  echo "Error: Docker build failed"
  exit 1
fi

echo ""
echo "Build successful!"
echo ""
echo "To push to Docker Hub, run:"
echo "  docker login"
echo "  docker push ${FULL_IMAGE_NAME}"
echo ""
echo "Or run this script with --push flag (add this option if needed)"
echo ""
echo "For others to use:"
echo "  docker pull ${FULL_IMAGE_NAME}"
echo "  docker run -it --gpus all ${FULL_IMAGE_NAME}"
