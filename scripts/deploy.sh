#!/bin/bash

echo "Deploying the application with Docker Compose..."

# 以后台模式启动服务
# -d: detach, 在后台运行
docker-compose up -d

echo "Application deployed. Use 'docker-compose logs -f' to see the logs."
