# API 使用示例和最佳实践

本文档提供 Local RAG 系统 API 的详细使用示例和最佳实践指南。

## 📋 API 概览

Local RAG 系统提供两个主要 API 端点：

| 端点 | 方法 | 功能 | 用途 |
|------|------|------|------|
| `/api/v1/ingest` | POST | 文档摄取 | 上传和处理文档 |
| `/api/v1/retrieve` | POST | 文档检索 | 查询相关文档片段 |

## 🚀 快速开始

### 启动服务

```bash
# 启动 API 服务
python start_server.py

# 验证服务状态
curl http://localhost:8000/health
```

### 查看 API 文档

访问 http://localhost:8000/docs 查看交互式 API 文档。

## 📤 文档摄取 API

### 基本用法

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "document_path": "documents/example.txt"
  }'
```

### 完整参数示例

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "document_path": "documents/technical_manual.txt",
    "chunk_size": 400,
    "chunk_overlap": 80
  }'
```

### Python 客户端示例

```python
import requests
import json
from pathlib import Path

class LocalRAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def ingest_document(self, document_path: str, chunk_size: int = 300, chunk_overlap: int = 50):
        """摄取单个文档"""
        url = f"{self.base_url}/api/v1/ingest"
        payload = {
            "document_path": document_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"摄取失败: {e}")
            return None
    
    def ingest_directory(self, directory_path: str, chunk_size: int = 300, chunk_overlap: int = 50):
        """批量摄取目录中的文档"""
        directory = Path(directory_path)
        results = []
        
        for file_path in directory.rglob("*.txt"):
            print(f"处理文件: {file_path}")
            result = self.ingest_document(str(file_path), chunk_size, chunk_overlap)
            if result:
                results.append(result)
        
        for file_path in directory.rglob("*.md"):
            print(f"处理文件: {file_path}")
            result = self.ingest_document(str(file_path), chunk_size, chunk_overlap)
            if result:
                results.append(result)
        
        return results

# 使用示例
client = LocalRAGClient()

# 摄取单个文档
result = client.ingest_document("documents/user_manual.txt")
if result:
    print(f"摄取成功: {result['chunks_count']} 个分片")

# 批量摄取
results = client.ingest_directory("documents/")
print(f"批量摄取完成: {len(results)} 个文件")
```

### JavaScript/Node.js 示例

```javascript
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

class LocalRAGClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.client = axios.create({
            baseURL,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    async ingestDocument(documentPath, chunkSize = 300, chunkOverlap = 50) {
        try {
            const response = await this.client.post('/api/v1/ingest', {
                document_path: documentPath,
                chunk_size: chunkSize,
                chunk_overlap: chunkOverlap
            });
            return response.data;
        } catch (error) {
            console.error('摄取失败:', error.response?.data || error.message);
            return null;
        }
    }

    async ingestDirectory(directoryPath, chunkSize = 300, chunkOverlap = 50) {
        const results = [];
        const files = await this.getTextFiles(directoryPath);
        
        for (const file of files) {
            console.log(`处理文件: ${file}`);
            const result = await this.ingestDocument(file, chunkSize, chunkOverlap);
            if (result) {
                results.push(result);
            }
        }
        
        return results;
    }

    async getTextFiles(dir) {
        const files = [];
        const items = await fs.readdir(dir, { withFileTypes: true });
        
        for (const item of items) {
            const fullPath = path.join(dir, item.name);
            if (item.isDirectory()) {
                files.push(...await this.getTextFiles(fullPath));
            } else if (item.name.endsWith('.txt') || item.name.endsWith('.md')) {
                files.push(fullPath);
            }
        }
        
        return files;
    }
}

// 使用示例
async function main() {
    const client = new LocalRAGClient();
    
    // 摄取单个文档
    const result = await client.ingestDocument('documents/example.txt');
    if (result) {
        console.log(`摄取成功: ${result.chunks_count} 个分片`);
    }
    
    // 批量摄取
    const results = await client.ingestDirectory('documents/');
    console.log(`批量摄取完成: ${results.length} 个文件`);
}

main().catch(console.error);
```

## 📥 文档检索 API

### 基本用法

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何配置系统参数？"
  }'
```

### 完整参数示例

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "机器学习模型训练的最佳实践",
    "retrieval_k": 15,
    "top_k": 5
  }'
```

### Python 检索客户端

```python
class LocalRAGClient:
    # ... (前面的代码)
    
    def retrieve(self, query: str, retrieval_k: int = 10, top_k: int = 3):
        """检索相关文档"""
        url = f"{self.base_url}/api/v1/retrieve"
        payload = {
            "query": query,
            "retrieval_k": retrieval_k,
            "top_k": top_k
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"检索失败: {e}")
            return None
    
    def search_and_display(self, query: str, retrieval_k: int = 10, top_k: int = 3):
        """检索并格式化显示结果"""
        results = self.retrieve(query, retrieval_k, top_k)
        
        if not results:
            print("检索失败")
            return
        
        print(f"查询: {query}")
        print(f"检索时间: {results.get('retrieval_time', 0):.3f}s")
        print("-" * 60)
        
        for i, result in enumerate(results.get('results', []), 1):
            print(f"结果 {i} (相关性: {result['score']:.3f})")
            print(f"内容: {result['content'][:200]}...")
            print(f"来源: {result['metadata']['file_name']}")
            print(f"分片: {result['metadata']['chunk_index']}")
            print("-" * 60)

# 使用示例
client = LocalRAGClient()

# 简单检索
results = client.retrieve("Python 编程技巧")

# 格式化显示
client.search_and_display("数据库优化方法", retrieval_k=15, top_k=5)
```

### 批量查询示例

```python
def batch_query(client, queries, output_file="query_results.json"):
    """批量查询并保存结果"""
    all_results = {}
    
    for query in queries:
        print(f"查询: {query}")
        results = client.retrieve(query)
        if results:
            all_results[query] = results
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"批量查询完成，结果保存到: {output_file}")
    return all_results

# 使用示例
queries = [
    "如何优化数据库性能？",
    "机器学习模型评估指标",
    "Python 异步编程最佳实践",
    "微服务架构设计原则"
]

client = LocalRAGClient()
batch_results = batch_query(client, queries)
```

## 🎯 最佳实践

### 1. 文档摄取最佳实践

#### 文档准备

```python
def prepare_documents(source_dir, target_dir):
    """文档预处理"""
    import shutil
    from pathlib import Path
    
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(exist_ok=True)
    
    for file_path in source.rglob("*"):
        if file_path.is_file():
            # 检查文件格式
            if file_path.suffix.lower() in ['.txt', '.md']:
                # 检查文件编码
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 清理内容
                    content = clean_text(content)
                    
                    # 保存到目标目录
                    target_file = target / file_path.name
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"处理完成: {file_path.name}")
                    
                except UnicodeDecodeError:
                    print(f"编码错误，跳过: {file_path}")

def clean_text(text):
    """文本清理"""
    import re
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】]', '', text)
    
    # 移除过短的行
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
    
    return '\n'.join(lines)
```

#### 分片策略选择

```python
def choose_chunk_params(document_type, content_length):
    """根据文档类型选择分片参数"""
    
    if document_type == "technical_manual":
        return {"chunk_size": 400, "chunk_overlap": 80}
    elif document_type == "conversation":
        return {"chunk_size": 200, "chunk_overlap": 40}
    elif document_type == "academic_paper":
        return {"chunk_size": 500, "chunk_overlap": 100}
    elif content_length < 1000:
        return {"chunk_size": 150, "chunk_overlap": 30}
    else:
        return {"chunk_size": 300, "chunk_overlap": 50}  # 默认值

# 使用示例
def smart_ingest(client, file_path, document_type="general"):
    """智能摄取"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    params = choose_chunk_params(document_type, len(content))
    return client.ingest_document(file_path, **params)
```

### 2. 检索优化最佳实践

#### 查询优化

```python
def optimize_query(query):
    """查询优化"""
    import jieba
    
    # 分词
    words = list(jieba.cut(query))
    
    # 移除停用词
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    words = [w for w in words if w not in stopwords and len(w) > 1]
    
    # 重新组合
    optimized_query = ' '.join(words)
    return optimized_query

def semantic_search(client, query, context_window=2):
    """语义搜索增强"""
    # 优化查询
    optimized_query = optimize_query(query)
    
    # 执行检索
    results = client.retrieve(optimized_query, retrieval_k=20, top_k=10)
    
    if not results:
        return None
    
    # 后处理：合并相邻分片
    enhanced_results = []
    for result in results['results']:
        # 获取上下文分片
        context = get_context_chunks(result, context_window)
        result['enhanced_content'] = context
        enhanced_results.append(result)
    
    return enhanced_results

def get_context_chunks(result, window_size):
    """获取上下文分片"""
    # 这里需要实现获取相邻分片的逻辑
    # 简化示例
    return result['content']
```

#### 结果后处理

```python
def post_process_results(results, query):
    """结果后处理"""
    if not results or 'results' not in results:
        return results
    
    processed_results = []
    
    for result in results['results']:
        # 计算查询词覆盖率
        coverage = calculate_query_coverage(result['content'], query)
        result['query_coverage'] = coverage
        
        # 添加摘要
        result['summary'] = generate_summary(result['content'])
        
        # 高亮关键词
        result['highlighted_content'] = highlight_keywords(result['content'], query)
        
        processed_results.append(result)
    
    # 重新排序
    processed_results.sort(key=lambda x: (x['score'] * 0.7 + x['query_coverage'] * 0.3), reverse=True)
    
    results['results'] = processed_results
    return results

def calculate_query_coverage(content, query):
    """计算查询词覆盖率"""
    import jieba
    
    query_words = set(jieba.cut(query))
    content_words = set(jieba.cut(content))
    
    if not query_words:
        return 0.0
    
    coverage = len(query_words & content_words) / len(query_words)
    return coverage

def generate_summary(content, max_length=100):
    """生成内容摘要"""
    if len(content) <= max_length:
        return content
    
    # 简单的摘要生成：取前面的句子
    sentences = content.split('。')
    summary = ""
    for sentence in sentences:
        if len(summary + sentence) <= max_length:
            summary += sentence + "。"
        else:
            break
    
    return summary.strip()

def highlight_keywords(content, query):
    """高亮关键词"""
    import jieba
    import re
    
    query_words = list(jieba.cut(query))
    highlighted = content
    
    for word in query_words:
        if len(word) > 1:  # 忽略单字符
            pattern = re.escape(word)
            highlighted = re.sub(pattern, f"**{word}**", highlighted)
    
    return highlighted
```

### 3. 性能优化

#### 连接池管理

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedRAGClient:
    def __init__(self, base_url="http://localhost:8000", max_retries=3):
        self.base_url = base_url
        self.session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # 配置连接池
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置超时
        self.session.timeout = 30

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
```

#### 并发处理

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def ingest_document_async(self, session, document_path, chunk_size=300, chunk_overlap=50):
        """异步文档摄取"""
        url = f"{self.base_url}/api/v1/ingest"
        payload = {
            "document_path": document_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        try:
            async with session.post(url, json=payload) as response:
                return await response.json()
        except Exception as e:
            print(f"摄取失败 {document_path}: {e}")
            return None
    
    async def batch_ingest_async(self, document_paths, chunk_size=300, chunk_overlap=50):
        """异步批量摄取"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.ingest_document_async(session, path, chunk_size, chunk_overlap)
                for path in document_paths
            ]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]

# 使用示例
async def main():
    client = AsyncRAGClient()
    document_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
    results = await client.batch_ingest_async(document_paths)
    print(f"批量摄取完成: {len(results)} 个文件")

# asyncio.run(main())
```

### 4. 错误处理和重试

```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1, backoff=2):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    
                    print(f"操作失败，{current_delay}秒后重试 ({retries}/{max_retries}): {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

class RobustRAGClient(LocalRAGClient):
    @retry_on_failure(max_retries=3, delay=1)
    def ingest_document(self, document_path, chunk_size=300, chunk_overlap=50):
        """带重试的文档摄取"""
        return super().ingest_document(document_path, chunk_size, chunk_overlap)
    
    @retry_on_failure(max_retries=3, delay=0.5)
    def retrieve(self, query, retrieval_k=10, top_k=3):
        """带重试的文档检索"""
        return super().retrieve(query, retrieval_k, top_k)
```

## 📊 性能监控

### 响应时间监控

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name):
    """计时器上下文管理器"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{operation_name} 耗时: {end_time - start_time:.3f}s")

# 使用示例
client = LocalRAGClient()

with timer("文档摄取"):
    result = client.ingest_document("large_document.txt")

with timer("文档检索"):
    results = client.retrieve("复杂查询问题")
```

### 批量性能测试

```python
def performance_test(client, test_queries, iterations=10):
    """性能测试"""
    import statistics
    
    response_times = []
    
    for i in range(iterations):
        for query in test_queries:
            start_time = time.time()
            result = client.retrieve(query)
            end_time = time.time()
            
            if result:
                response_times.append(end_time - start_time)
    
    if response_times:
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"性能测试结果 ({len(response_times)} 次请求):")
        print(f"  平均响应时间: {avg_time:.3f}s")
        print(f"  中位数响应时间: {median_time:.3f}s")
        print(f"  最大响应时间: {max_time:.3f}s")
        print(f"  最小响应时间: {min_time:.3f}s")

# 使用示例
test_queries = [
    "系统配置方法",
    "性能优化技巧",
    "错误处理策略"
]

client = LocalRAGClient()
performance_test(client, test_queries)
```

## 🔧 故障排除

### 常见 API 错误

```python
def handle_api_errors(response):
    """处理 API 错误"""
    if response.status_code == 200:
        return response.json()
    
    error_messages = {
        400: "请求参数错误",
        404: "文档未找到",
        422: "参数验证失败",
        500: "服务器内部错误"
    }
    
    error_msg = error_messages.get(response.status_code, f"未知错误 ({response.status_code})")
    
    try:
        error_detail = response.json()
        print(f"API 错误: {error_msg}")
        print(f"详细信息: {error_detail}")
    except:
        print(f"API 错误: {error_msg}")
        print(f"响应内容: {response.text}")
    
    return None

# 在客户端中使用
class SafeRAGClient(LocalRAGClient):
    def _make_request(self, method, url, **kwargs):
        """安全的请求方法"""
        try:
            response = self.session.request(method, url, **kwargs)
            return handle_api_errors(response)
        except requests.exceptions.ConnectionError:
            print("连接错误: 无法连接到服务器，请检查服务是否启动")
            return None
        except requests.exceptions.Timeout:
            print("超时错误: 请求超时，请稍后重试")
            return None
        except Exception as e:
            print(f"未知错误: {e}")
            return None
```

---

**提示**: 这些示例展示了 Local RAG 系统 API 的各种使用方式。根据具体需求选择合适的方法，并注意错误处理和性能优化。