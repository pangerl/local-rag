# 使用官方 Python 镜像作为基础镜像
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
# --no-cache-dir: 不缓存包，减小镜像体积
# --prefer-binary: 优先使用二进制包，加快安装速度
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动应用
# 使用 uvicorn 启动 FastAPI 应用
# --host 0.0.0.0: 监听所有网络接口，允许外部访问
# --port 8000: 指定服务端口
# app.main:main: 指定 FastAPI 应用实例的位置 (app/main.py 文件中的 main 对象)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
