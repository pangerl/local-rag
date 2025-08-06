# API 使用示例和最佳实践

本文档提供 Local RAG 系统 API 的详细使用示例和最佳实践指南。

## 📋 API 概览

Local RAG 系统提供以下几类 API 端点：

| 类别 | 端点前缀 | 主要功能 |
|:--- |:--- |:--- |
| **文档管理** | `/api/v1/` | 文档的摄取、上传、查询和删除 |
| **系统监控** | `/api/v1/monitoring/` | 获取系统性能指标和资源状态 |
| **Web 界面** | `/admin` | 提供图形化的管理和查询界面 |

## 🚀 快速开始

### 启动服务

```bash
# 启动 API 服务
python start_server.py

# 验证服务状态
curl http://localhost:8000/api/v1/health
```

### 查看 API 文档

访问 http://localhost:8000/docs 查看完整的交互式 API 文档。

---

## 📦 文档管理 API

### 1. 文档摄取 (文件上传)

**端点**: `POST /api/v1/ingest`

此接口用于直接上传文档文件进行处理，支持多种格式。

**cURL 示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf" \
  -F "chunk_size=300" \
  -F "chunk_overlap=50"
```

### 2. 批量摄取 (基于目录)

**端点**: `POST /api/v1/ingest/load`

此接口用于从服务器的本地目录批量加载和处理文档。

**cURL 示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/load" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "my_documents",
    "chunk_size": 400,
    "chunk_overlap": 80
  }'
```

**Python 客户端示例**:

```python
import requests

def upload_document(file_path: str):
    """上传并摄取文档"""
    url = "http://localhost:8000/api/v1/ingest"
    with open(file_path, "rb") as f:
        files = {"file": (f.name, f, "application/octet-stream")}
        data = {"chunk_size": 300, "chunk_overlap": 50}

        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"上传失败: {e}")
            return None

# 使用示例
result = upload_document("path/to/your/report.docx")
if result:
    print(f"上传成功: {result['message']}")
```

### 4. 文档检索

**端点**: `POST /api/v1/retrieve`

根据查询文本检索相关的文档片段。

**cURL 示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "机器学习模型训练的最佳实践",
    "retrieval_k": 15,
    "top_k": 5,
    "use_reranker": true
  }'
```

### 5. 获取所有文档信息

**端点**: `GET /api/v1/documents`

返回已处理的所有文档的列表及其详细信息。

**cURL 示例**:

```bash
curl -X GET "http://localhost:8000/api/v1/documents"
```

**响应示例**:
```json
[
  {
    "document_path": "docs/doc1.txt",
    "chunk_count": 20,
    "created_at": "2024-07-29T10:00:00",
    "text_length": 15000,
    "chunk_size": 300,
    "file_size": 15360
  }
]
```

### 6. 删除文档

**端点**: `DELETE /api/v1/documents/{document_path}`

删除指定的文档及其所有分片数据。

**cURL 示例**:

```bash
# URL 编码路径
curl -X DELETE "http://localhost:8000/api/v1/documents/docs%2Ftechnical_manual.txt"
```

**Python 客户端示例**:

```python
import requests
from urllib.parse import quote

def delete_document(document_path: str):
    """删除指定文档"""
    encoded_path = quote(document_path, safe='')
    url = f"http://localhost:8000/api/v1/documents/{encoded_path}"

    try:
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"删除失败: {e}")
        return None

# 使用示例
result = delete_document("docs/technical_manual.txt")
if result:
    print(f"删除成功: {result['message']}")
```

---

## ⚙️ 系统与监控 API

### 1. 健康检查

**端点**: `GET /api/v1/health`

检查系统各组件的健康状态。

**cURL 示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

### 2. 系统统计

**端点**: `GET /api/v1/stats`

获取系统运行统计信息，如文档总数、分片总数等。

**cURL 示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/stats"
```

### 3. 获取系统指标

**端点**: `GET /api/v1/monitoring/metrics`

获取指定时间窗口内的系统性能指标摘要。

**cURL 示例**:
```bash
# 获取最近 5 分钟的指标
curl -X GET "http://localhost:8000/api/v1/monitoring/metrics?time_window=300"
```

### 4. 获取系统状态

**端点**: `GET /api/v1/monitoring/system`

获取当前系统资源使用情况和进程状态。

**cURL 示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/monitoring/system"
```

---

## 🎯 最佳实践

### 1. 完整的 Python 客户端

下面是一个更完整的 Python 客户端，封装了大部分 API 调用。

```python
import requests
import json
from pathlib import Path
from urllib.parse import quote

class LocalRAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def _request(self, method, endpoint, **kwargs):
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API 请求失败 ({method} {url}): {e}")
            if e.response is not None:
                print(f"响应内容: {e.response.text}")
            return None

    def upload_document(self, file_path: str, chunk_size: int = 300, chunk_overlap: int = 50):
        """上传并摄取文档"""
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            data = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            return self._request("POST", "/api/v1/ingest", files=files, data=data)

    def ingest_load(self, path: str, chunk_size: int = 300, chunk_overlap: int = 50):
        """从目录批量摄取文档"""
        payload = {
            "path": path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        return self._request("POST", "/api/v1/ingest/load", json=payload)

    def retrieve(self, query: str, retrieval_k: int = 10, top_k: int = 3, use_reranker: bool = True):
        """检索相关文档"""
        payload = {
            "query": query,
            "retrieval_k": retrieval_k,
            "top_k": top_k,
            "use_reranker": use_reranker
        }
        return self._request("POST", "/api/v1/retrieve", json=payload)

    def list_documents(self):
        """获取所有文档信息"""
        return self._request("GET", "/api/v1/documents")

    def delete_document(self, document_path: str):
        """删除指定文档"""
        encoded_path = quote(document_path, safe='')
        return self._request("DELETE", f"/api/v1/documents/{encoded_path}")

    def get_health(self):
        """获取系统健康状态"""
        return self._request("GET", "/api/v1/health")

    def get_stats(self):
        """获取系统统计信息"""
        return self._request("GET", "/api/v1/stats")

# 使用示例
client = LocalRAGClient()

# 1. 上传文档
upload_result = client.upload_document("path/to/your/document.txt")
if upload_result:
    print("上传成功:", upload_result["message"])

# 2. 检索
search_results = client.retrieve("你的查询问题")
if search_results:
    for res in search_results.get("results", []):
        print(f"  - {res['text'][:80]}... (Score: {res['similarity_score']:.2f})")

# 3. 查看所有文档
docs = client.list_documents()
if docs:
    print(f"当前共有 {len(docs)} 个文档。")

# 4. 删除文档
delete_result = client.delete_document("your/document.txt")
if delete_result:
    print("删除成功:", delete_result["message"])

# 5. 查看系统状态
health = client.get_health()
if health:
    print("系统状态:", health["status"])
```

### 2. 性能与并发

对于大批量的数据处理，建议使用异步客户端或多线程来提高效率。

```python
import asyncio
import aiohttp
from pathlib import Path

class AsyncRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    async def upload_document(self, session, file_path):
        url = f"{self.base_url}/api/v1/ingest"
        data = aiohttp.FormData()
        data.add_field('file',
                       open(file_path, 'rb'),
                       filename=Path(file_path).name,
                       content_type='application/octet-stream')

        try:
            async with session.post(url, data=data) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            print(f"上传失败 {file_path}: {e}")
            return None

    async def batch_upload(self, directory_path: str):
        """异步批量上传"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for file_path in Path(directory_path).rglob("*.*"):
                if file_path.is_file():
                    tasks.append(self.upload_document(session, file_path))

            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]

# 使用示例
async def main():
    client = AsyncRAGClient()
    results = await client.batch_upload("path/to/your/docs_folder")
    print(f"批量上传完成: {len(results)} 个文件")

# 在你的异步环境中运行:
# asyncio.run(main())
```

### 3. 错误处理和重试

生产环境中，建议在客户端实现重试逻辑，特别是对于网络不稳定或服务器可能临时过载的情况。

```python
import time
from functools import wraps

def retry(max_tries=3, delay=1, backoff=2):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    tries += 1
                    if tries == max_tries:
                        raise
                    print(f"请求失败: {e}, {delay}s 后重试...")
                    time.sleep(delay)
                    delay *= backoff
            return None
        return wrapper
    return decorator

class RobustRAGClient(LocalRAGClient):
    @retry()
    def retrieve(self, *args, **kwargs):
        return super().retrieve(*args, **kwargs)

# 使用示例
robust_client = RobustRAGClient()
results = robust_client.retrieve("一个重要的查询")
```

---

**提示**: 这些示例展示了 Local RAG 系统 API 的各种使用方式。根据具体需求选择合适的方法，并注意错误处理和性能优化。
