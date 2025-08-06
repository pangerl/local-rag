# Local RAG 系统

一个轻量级、纯本地运行的检索增强生成（RAG）知识库系统，专为中文文本优化，支持完全离线部署。

## 🌟 特性

- **完全离线运行**: 无需网络连接，确保数据隐私和安全
- **中文优化**: 使用 jieba 分词和中文优化的嵌入模型
- **高效检索**: 基于向量相似性搜索和重排序的双重检索机制
- **RESTful API**: 提供标准的 HTTP API 接口
- **批量处理**: 支持命令行批量文档导入
- **灵活配置**: 支持多种配置方式和参数调整

## 📋 系统要求

### 硬件要求

- **内存**: 建议 8GB 以上（模型加载需要约 2-3GB）
- **存储**: 至少 5GB 可用空间（模型文件约 2-3GB）
- **CPU**: 支持 AVX 指令集的现代 CPU

### 软件要求

- **Python**: 3.13 或更高版本
- **操作系统**: Linux、macOS 或 Windows
- **包管理器**: uv（推荐）或 pip

## 🚀 快速开始

### 1. 环境准备

#### 安装 uv（推荐）

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 或使用 pip

```bash
pip install uv
```

### 2. 项目安装

```bash
# 克隆项目
git clone <repository-url>
cd local-rag

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
uv pip install -r requirements.txt
```

### 3. 模型下载和部署

#### 自动下载脚本（推荐）

使用模型下载脚本[download_models.py](scripts/download_models.py)，下载模型文件，并将 models 文件放到对应的数据目录下，一般为项目目录的 data 文件夹。

### 4. 启动服务

依赖 `libmagic` 库，根据本地环境自行安装。

```bash
# 使用启动脚本（推荐）
python start_server.py

# 或直接启动
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，访问 http://localhost:8000/docs 查看 API 文档。


## ⚙️ 配置说明

### 环境变量配置

项目通过 `.env` 文件进行配置。为了方便起步，您可以复制 `.env.template` 文件来创建自己的配置文件：

```bash
cp .env.template .env
```

然后根据需要编辑 `.env` 文件。

## 🖥️ 管理界面

本系统内置了一个基于 Web 的管理界面，方便用户进行文档管理和检索测试。

![管理页面](docs/img/admin.png "页面示例")

- **管理页面**: [http://localhost:8000/admin](http://localhost:8000/admin)
- **检索页面**: [http://localhost:8000/admin/search](http://localhost:8000/admin/search)


## 📖 API 使用指南

本系统提供了一套完整的 RESTful API 用于文档管理和系统监控。所有 API 端点都支持标准的 HTTP 方法，并返回 JSON 格式的响应。

### 主要 API 功能

- **文档管理**:
  - `POST /api/v1/ingest`: 从本地路径摄取文档。
  - `POST /api/v1/ingest/upload`: 通过文件上传方式摄取文档。
  - `POST /api/v1/ingest/load`: 从本地目录批量摄取文档。
  - `POST /api/v1/retrieve`: 根据查询检索文档片段。
  - `GET /api/v1/documents`: 获取所有已处理的文档列表。
  - `DELETE /api/v1/documents/{document_path}`: 删除指定的文档。
- **系统与监控**:
  - `GET /api/v1/health`: 检查系统健康状态。
  - `GET /api/v1/stats`: 获取系统运行统计数据。
  - `GET /api/v1/monitoring/metrics`: 获取详细的性能指标。
  - `GET /api/v1/monitoring/system`: 获取系统资源使用情况。

### 交互式文档

要查看所有 API 的详细参数、响应模型和在线测试工具，请访问服务启动后自动生成的交互式文档：

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 详细使用示例

我们提供了详细的 API 使用示例和客户端代码，包括 cURL、Python 和 JavaScript。这些示例可以帮助您快速集成和使用本系统。

请参阅 [**API 使用示例和最佳实践 (docs/API_EXAMPLES.md)**](docs/API_EXAMPLES.md) 获取完整指南。

### 功能概览

- **文档列表**: 查看所有已入库的文档，包括路径、分片数量、大小等信息。
- **文件上传**: 直接通过浏览器上传一个或多个文档文件进行处理。
- **文档删除**: 在列表中方便地删除不再需要的文档。
- **实时检索**: 在检索页面输入查询，调整参数（如 Top K、是否使用重排器），并立即看到结果。
- **系统监控**: （规划中）集成系统健康状态和关键指标的可视化图表。

这个界面对于快速验证、内容管理和功能演示非常有用，无需编写任何代码即可与系统的核心功能进行交互。

## 🔧 批量导入工具

### 基本用法

```bash
# 导入单个文件
python scripts/bulk_ingest.py --path documents/example.txt

# 导入整个目录
python scripts/bulk_ingest.py --path documents/

# 自定义分片参数
python scripts/bulk_ingest.py \
  --path documents/ \
  --chunk-size 500 \
  --chunk-overlap 100
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--path` | 文档路径（文件或目录） | 必需 |
| `--chunk-size` | 分片大小（词元数） | 300 |
| `--chunk-overlap` | 分片重叠（词元数） | 50 |

### 批量导入示例

**脚本导入**

```bash
# 准备文档目录
mkdir -p documents
echo "这是一个测试文档，用于演示批量导入功能。" > documents/test1.txt
echo "另一个测试文档，包含不同的内容。" > documents/test2.txt

# 执行批量导入
python scripts/bulk_ingest.py --path documents/

# 验证导入结果
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "测试文档"}'
```

**接口导入**

确保path目录位于数据目录下

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/load" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "my_documents",
    "chunk_size": 400,
    "chunk_overlap": 80
  }'
```

### 日志分析

#### 查看日志文件

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/error.log

# 查看性能日志
tail -f logs/performance.log
```

#### 日志级别调整

```bash
# 启用调试日志
LOG_LEVEL=DEBUG python start_server.py

# 查看详细的处理过程
grep "DocumentProcessor" logs/app.log
```

### 性能优化

#### 1. 调整分片参数

```python
# 较小的分片 - 更精确但检索较慢
chunk_size = 200
chunk_overlap = 30

# 较大的分片 - 检索较快但可能不够精确
chunk_size = 500
chunk_overlap = 100
```

#### 2. 调整检索参数

```python
# 快速检索 - 较少候选文档
retrieval_k = 5
top_k = 2

# 精确检索 - 更多候选文档
retrieval_k = 20
top_k = 5
```

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 🆘 获取帮助

- **问题报告**: 在 GitHub Issues 中提交问题
- **功能请求**: 在 GitHub Issues 中提交功能请求
- **文档问题**: 在项目 Wiki 中查找更多信息

---

**注意**: 本系统设计为完全离线运行，请确保在部署前下载所有必需的模型文件。
