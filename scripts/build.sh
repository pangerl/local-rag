#!/bin/bash

# 设置镜像名称和标签
IMAGE_NAME="local-rag"
IMAGE_TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"

# 使用 Docker Compose 构建镜像
# --no-cache: 强制重新构建，不使用缓存
docker-compose build --no-cache

echo "Image build complete."
