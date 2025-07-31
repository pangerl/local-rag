# 使用官方 Python 镜像作为基础镜像
FROM python:3.13-slim

# 设置环境变量：不生成 .pyc、日志直接输出控制台
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --prefer-binary -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 使用 ENTRYPOINT + CMD，增强可维护性
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]