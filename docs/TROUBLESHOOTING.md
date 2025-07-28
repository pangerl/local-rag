# 故障排除和常见问题解答

本文档提供 Local RAG 系统常见问题的解决方案和故障排除指南。

## 🚨 紧急故障处理

### 系统无法启动

**症状**: 运行 `python start_server.py` 后系统崩溃或无响应

**快速诊断**:
```bash
# 检查 Python 版本
python --version

# 检查依赖安装
pip list | grep -E "(fastapi|torch|transformers|chromadb)"

# 检查模型文件
ls -la models/

# 查看错误日志
tail -n 50 logs/app.log
```

**常见原因和解决方案**:

1. **Python 版本不兼容**
   ```bash
   # 确保使用 Python 3.13+
   python3.13 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **模型文件缺失**
   ```bash
   # 重新下载模型
   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5')
   snapshot_download('BAAI/bge-reranker-base', local_dir='models/bge-reranker-base')
   "
   ```

3. **端口被占用**
   ```bash
   # 查找占用进程
   lsof -i :8000
   
   # 杀死进程
   kill -9 <PID>
   
   # 或使用不同端口
   API_PORT=8001 python start_server.py
   ```

## 🔧 安装和配置问题

### Q1: 依赖安装失败

**问题**: `pip install -r requirements.txt` 失败

**解决方案**:
```bash
# 升级 pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 分步安装关键依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
pip install fastapi uvicorn chromadb jieba

# 如果仍然失败，使用 conda
conda create -n local-rag python=3.13
conda activate local-rag
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements.txt
```

### Q2: 模型下载速度慢或失败

**问题**: 模型下载超时或速度极慢

**解决方案**:
```bash
# 方法1: 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
python download_models.py

# 方法2: 使用代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 方法3: 手动下载
# 访问 https://hf-mirror.com/BAAI/bge-small-zh-v1.5
# 下载所有文件到 models/bge-small-zh-v1.5/

# 方法4: 使用 git-lfs
git lfs install
cd models
git clone https://hf-mirror.com/BAAI/bge-small-zh-v1.5
git clone https://hf-mirror.com/BAAI/bge-reranker-base
```

### Q3: 权限错误

**问题**: `Permission denied` 错误

**解决方案**:
```bash
# 检查目录权限
ls -la

# 修复权限
chmod -R 755 models/ data/ logs/
chown -R $USER:$USER models/ data/ logs/

# 如果是 Windows，以管理员身份运行
# 或检查防病毒软件是否阻止文件访问
```

## 🧠 模型相关问题

### Q4: 模型加载失败

**错误信息**: `ModelLoadError: 无法加载嵌入模型`

**诊断步骤**:
```bash
# 1. 检查模型文件完整性
python -c "
import os
from pathlib import Path

model_dir = Path('models/bge-small-zh-v1.5')
required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']

for file in required_files:
    file_path = model_dir / file
    if file_path.exists():
        print(f'✅ {file}: {file_path.stat().st_size} bytes')
    else:
        print(f'❌ {file}: 文件不存在')
"

# 2. 测试模型加载
python -c "
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer('models/bge-small-zh-v1.5')
    print('✅ 模型加载成功')
    
    # 测试编码
    embeddings = model.encode(['测试文本'])
    print(f'✅ 编码测试成功，维度: {embeddings.shape}')
except Exception as e:
    print(f'❌ 模型加载失败: {e}')
"
```

**解决方案**:
```bash
# 1. 重新下载模型
rm -rf models/bge-small-zh-v1.5
rm -rf models/bge-reranker-base
python download_models.py

# 2. 检查磁盘空间
df -h

# 3. 检查内存
free -h

# 4. 使用更小的模型（如果内存不足）
# 修改 app/core/config.py 中的模型路径
```

### Q5: 内存不足错误

**错误信息**: `OutOfMemoryError` 或系统卡顿

**解决方案**:
```bash
# 1. 检查系统内存
free -h
htop

# 2. 减少模型内存占用
# 在 app/core/config.py 中添加：
# TORCH_DEVICE = "cpu"  # 强制使用 CPU
# MODEL_CACHE_SIZE = 1  # 减少缓存

# 3. 调整分片参数
# 减少 DEFAULT_CHUNK_SIZE 和 DEFAULT_RETRIEVAL_K

# 4. 使用更小的模型
# 下载 bge-small-zh 而不是 bge-base-zh

# 5. 增加虚拟内存（Linux）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 💾 数据库问题

### Q6: ChromaDB 连接失败

**错误信息**: `DatabaseError: ChromaDB 连接失败`

**解决方案**:
```bash
# 1. 检查数据目录
ls -la data/chroma_db/

# 2. 清理损坏的数据库
rm -rf data/chroma_db/*
mkdir -p data/chroma_db

# 3. 检查权限
chmod 755 data/chroma_db

# 4. 测试数据库连接
python -c "
import chromadb
try:
    client = chromadb.PersistentClient(path='data/chroma_db')
    collection = client.get_or_create_collection('test')
    print('✅ ChromaDB 连接成功')
except Exception as e:
    print(f'❌ ChromaDB 连接失败: {e}')
"

# 5. 如果仍然失败，使用内存数据库
# 在配置中设置 CHROMA_DB_PATH = ":memory:"
```

### Q7: 数据库文件损坏

**症状**: 查询返回空结果或异常错误

**解决方案**:
```bash
# 1. 备份现有数据
cp -r data/chroma_db data/chroma_db_backup_$(date +%Y%m%d)

# 2. 检查数据库完整性
python -c "
import chromadb
import sqlite3

# 检查 SQLite 文件
db_path = 'data/chroma_db/chroma.sqlite3'
try:
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA integrity_check;')
    print('✅ 数据库文件完整')
    conn.close()
except Exception as e:
    print(f'❌ 数据库文件损坏: {e}')
"

# 3. 重建数据库
rm -rf data/chroma_db
mkdir -p data/chroma_db

# 4. 重新导入文档
python scripts/bulk_ingest.py --path documents/
```

## 🌐 API 问题

### Q8: API 请求超时

**错误信息**: `Request timeout` 或 `Connection timeout`

**解决方案**:
```bash
# 1. 检查服务状态
curl -I http://localhost:8000/health

# 2. 增加超时时间
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  --max-time 60 \
  -H "Content-Type: application/json" \
  -d '{"query": "测试查询"}'

# 3. 检查系统负载
top
iostat 1

# 4. 优化查询参数
# 减少 retrieval_k 和 top_k 参数

# 5. 在客户端代码中设置超时
import requests
requests.post(url, json=data, timeout=60)
```

### Q9: API 返回 422 错误

**错误信息**: `Unprocessable Entity`

**解决方案**:
```bash
# 1. 检查请求格式
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "document_path": "/absolute/path/to/document.txt",
    "chunk_size": 300,
    "chunk_overlap": 50
  }' -v

# 2. 验证参数类型
# chunk_size 和 chunk_overlap 必须是整数
# document_path 必须是字符串

# 3. 检查文件路径
# 确保文件存在且可读
ls -la /path/to/document.txt

# 4. 查看详细错误信息
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}' | jq .
```

### Q10: 文档格式不支持

**错误信息**: `ValidationError: 不支持的文件格式`

**解决方案**:
```bash
# 1. 检查支持的格式
python -c "
from app.core.config import settings
print('支持的格式:', settings.SUPPORTED_FORMATS)
"

# 2. 转换文档格式
# Word 转 txt
pandoc document.docx -o document.txt

# PDF 转 txt
pdftotext document.pdf document.txt

# HTML 转 txt
pandoc document.html -o document.txt

# 3. 批量转换
find documents/ -name "*.docx" -exec pandoc {} -o {}.txt \;

# 4. 检查文件编码
file -i document.txt
# 如果不是 UTF-8，转换编码
iconv -f GBK -t UTF-8 document.txt > document_utf8.txt
```

## 📊 性能问题

### Q11: 检索速度慢

**症状**: 查询响应时间超过 5 秒

**优化方案**:
```bash
# 1. 调整检索参数
# 减少候选文档数量
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "测试查询",
    "retrieval_k": 5,
    "top_k": 2
  }'

# 2. 优化分片大小
# 使用更大的分片减少总数量
python scripts/bulk_ingest.py --path documents/ --chunk-size 500

# 3. 检查系统资源
htop
iotop

# 4. 使用 SSD 存储
# 将 data/ 目录移动到 SSD

# 5. 启用模型缓存
# 在配置中设置适当的缓存大小
```

### Q12: 内存使用过高

**症状**: 系统内存占用持续增长

**解决方案**:
```bash
# 1. 监控内存使用
python -c "
import psutil
import os

process = psutil.Process(os.getpid())
print(f'内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# 2. 重启服务释放内存
pkill -f "python start_server.py"
python start_server.py

# 3. 调整配置减少内存使用
# 在 app/core/config.py 中：
# DEFAULT_RETRIEVAL_K = 5  # 减少检索数量
# DEFAULT_CHUNK_SIZE = 200  # 减少分片大小

# 4. 使用内存分析工具
pip install memory-profiler
python -m memory_profiler start_server.py

# 5. 启用垃圾回收
python -c "
import gc
gc.collect()
print('垃圾回收完成')
"
```

## 🔍 调试和日志

### Q13: 如何启用详细日志

**解决方案**:
```bash
# 1. 设置日志级别
LOG_LEVEL=DEBUG python start_server.py

# 2. 查看不同类型的日志
tail -f logs/app.log        # 应用日志
tail -f logs/error.log      # 错误日志
tail -f logs/performance.log # 性能日志

# 3. 过滤特定组件的日志
grep "DocumentProcessor" logs/app.log
grep "VectorRetriever" logs/app.log
grep "ModelLoader" logs/app.log

# 4. 实时监控日志
tail -f logs/app.log | grep -E "(ERROR|WARNING)"
```

### Q14: 如何调试 API 请求

**解决方案**:
```bash
# 1. 启用请求追踪
# 在日志中查看请求 ID
grep "request_id" logs/app.log

# 2. 使用 curl 详细模式
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "测试"}' \
  -v

# 3. 检查 API 文档
# 访问 http://localhost:8000/docs

# 4. 使用 Python 调试
import requests
import logging

logging.basicConfig(level=logging.DEBUG)
response = requests.post(url, json=data)
```

## 🔄 数据迁移和备份

### Q15: 如何备份和恢复数据

**备份方案**:
```bash
# 1. 完整备份
tar -czf local_rag_backup_$(date +%Y%m%d).tar.gz \
  data/ models/ logs/ app/core/config.py

# 2. 仅备份数据库
cp -r data/chroma_db data/chroma_db_backup_$(date +%Y%m%d)

# 3. 导出文档列表
python -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma_db')
collection = client.get_collection('documents')
metadata = collection.get(include=['metadatas'])
print('文档数量:', len(metadata['metadatas']))
"

# 4. 自动备份脚本
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
cp -r data/chroma_db $BACKUP_DIR/
echo "备份完成: $BACKUP_DIR"
EOF
chmod +x backup.sh
```

**恢复方案**:
```bash
# 1. 恢复数据库
rm -rf data/chroma_db
cp -r data/chroma_db_backup_20240101 data/chroma_db

# 2. 验证恢复
python -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma_db')
collections = client.list_collections()
print('集合数量:', len(collections))
"

# 3. 重新索引（如果需要）
python scripts/bulk_ingest.py --path documents/
```

## 🚀 性能调优

### 系统级优化

```bash
# 1. 调整文件描述符限制
ulimit -n 65536

# 2. 优化 Python GC
export PYTHONHASHSEED=0
export PYTHONOPTIMIZE=1

# 3. 使用更快的 JSON 库
pip install orjson
# 在代码中替换 json 为 orjson

# 4. 启用 JIT 编译
pip install numba
# 在计算密集型函数上使用 @numba.jit

# 5. 使用多进程
# 启动多个 worker
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### 应用级优化

```python
# 1. 连接池配置
# 在 app/core/config.py 中：
CHROMA_POOL_SIZE = 10
MODEL_CACHE_SIZE = 100

# 2. 批量处理
# 批量摄取文档而不是逐个处理

# 3. 异步处理
# 使用 asyncio 处理 I/O 密集型操作

# 4. 缓存策略
# 缓存常用查询结果
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieve(query_hash):
    # 实现缓存逻辑
    pass
```

## 📞 获取帮助

### 社区支持

- **GitHub Issues**: 报告 bug 和功能请求
- **讨论区**: 技术讨论和经验分享
- **文档**: 查看最新文档和教程

### 提交问题时请包含

1. **系统信息**:
   ```bash
   python --version
   pip list | grep -E "(torch|transformers|chromadb)"
   uname -a
   ```

2. **错误日志**:
   ```bash
   tail -n 100 logs/app.log
   tail -n 50 logs/error.log
   ```

3. **配置信息**:
   ```bash
   python -c "
   from app.core.config import settings
   print('模型路径:', settings.embedding_model_path)
   print('数据库路径:', settings.chroma_db_full_path)
   "
   ```

4. **重现步骤**: 详细描述如何重现问题

### 紧急联系

对于生产环境的紧急问题：

1. 立即停止服务：`pkill -f "python start_server.py"`
2. 备份数据：`cp -r data/chroma_db data/emergency_backup`
3. 查看错误日志：`tail -n 200 logs/error.log`
4. 联系技术支持并提供完整的错误信息

---

**提示**: 大多数问题都可以通过检查日志文件和验证配置来解决。遇到问题时，首先查看 `logs/app.log` 和 `logs/error.log` 文件。