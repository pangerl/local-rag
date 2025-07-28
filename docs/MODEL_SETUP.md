# 模型下载和部署指南

本指南详细说明如何下载和部署 Local RAG 系统所需的模型文件。

## 📋 模型概览

Local RAG 系统使用两个预训练模型：

| 模型 | 用途 | 大小 | 来源 |
|------|------|------|------|
| bge-small-zh-v1.5 | 文本嵌入 | ~400MB | BAAI/bge-small-zh-v1.5 |
| bge-reranker-base | 结果重排序 | ~1.1GB | BAAI/bge-reranker-base |

## 🚀 快速部署

### 方法一：自动下载脚本（推荐）

创建并运行以下脚本：

```bash
# 创建 download_models.py
cat > download_models.py << 'EOF'
#!/usr/bin/env python3
"""
Local RAG 模型自动下载脚本
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(repo_id: str, local_dir: Path, description: str):
    """下载单个模型"""
    try:
        logger.info(f"开始下载 {description}...")
        logger.info(f"仓库: {repo_id}")
        logger.info(f"目标目录: {local_dir}")
        
        # 创建目录
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        logger.info(f"✅ {description} 下载完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ {description} 下载失败: {e}")
        return False

def verify_model(model_dir: Path, model_name: str):
    """验证模型文件完整性"""
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.warning(f"⚠️  {model_name} 缺少文件: {missing_files}")
        return False
    else:
        logger.info(f"✅ {model_name} 文件完整")
        return True

def main():
    """主函数"""
    logger.info("开始下载 Local RAG 系统模型...")
    
    # 模型配置
    models = [
        {
            "repo_id": "BAAI/bge-small-zh-v1.5",
            "local_dir": Path("models/bge-small-zh-v1.5"),
            "description": "中文嵌入模型 (bge-small-zh-v1.5)"
        },
        {
            "repo_id": "BAAI/bge-reranker-base",
            "local_dir": Path("models/bge-reranker-base"),
            "description": "重排序模型 (bge-reranker-base)"
        }
    ]
    
    # 检查网络连接
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code != 200:
            logger.error("❌ 无法连接到 Hugging Face，请检查网络连接")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 网络连接检查失败: {e}")
        sys.exit(1)
    
    # 下载模型
    success_count = 0
    for model in models:
        if download_model(model["repo_id"], model["local_dir"], model["description"]):
            if verify_model(model["local_dir"], model["description"]):
                success_count += 1
    
    # 总结
    logger.info(f"模型下载完成: {success_count}/{len(models)} 成功")
    
    if success_count == len(models):
        logger.info("🎉 所有模型下载成功！系统已准备就绪。")
        
        # 显示模型信息
        logger.info("\n📊 模型信息:")
        for model in models:
            model_dir = model["local_dir"]
            if model_dir.exists():
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                logger.info(f"  - {model['description']}: {size_mb:.1f} MB")
    else:
        logger.error("❌ 部分模型下载失败，请检查错误信息并重试")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# 安装依赖并运行
pip install huggingface_hub requests
python download_models.py
```

### 方法二：使用 huggingface-cli

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载嵌入模型
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5

# 下载重排序模型
huggingface-cli download BAAI/bge-reranker-base --local-dir models/bge-reranker-base
```

### 方法三：使用 git-lfs

```bash
# 安装 git-lfs
git lfs install

# 创建模型目录
mkdir -p models

# 克隆嵌入模型
cd models
git clone https://huggingface.co/BAAI/bge-small-zh-v1.5
git clone https://huggingface.co/BAAI/bge-reranker-base
cd ..
```

## 🔧 手动下载

如果自动下载失败，可以手动下载模型文件：

### 1. 嵌入模型 (bge-small-zh-v1.5)

访问 https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/main

下载以下文件到 `models/bge-small-zh-v1.5/` 目录：

```text
├── config.json
├── pytorch_model.bin
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

### 2. 重排序模型 (bge-reranker-base)

访问 https://huggingface.co/BAAI/bge-reranker-base/tree/main

下载以下文件到 `models/bge-reranker-base/` 目录：

```text
├── config.json
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

## ✅ 验证安装

### 1. 检查文件结构

```bash
# 检查目录结构
tree models/

# 预期输出：
# models/
# ├── bge-small-zh-v1.5/
# │   ├── config.json
# │   ├── pytorch_model.bin
# │   ├── sentence_bert_config.json
# │   ├── special_tokens_map.json
# │   ├── tokenizer.json
# │   ├── tokenizer_config.json
# │   └── vocab.txt
# └── bge-reranker-base/
#     ├── config.json
#     ├── pytorch_model.bin
#     ├── special_tokens_map.json
#     ├── tokenizer.json
#     ├── tokenizer_config.json
#     └── vocab.txt
```

### 2. 验证模型加载

```bash
# 创建验证脚本
cat > verify_models.py << 'EOF'
#!/usr/bin/env python3
"""
模型验证脚本
"""

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

def test_model(model_path: str, model_name: str):
    """测试模型加载"""
    try:
        print(f"测试 {model_name}...")
        model = SentenceTransformer(model_path)
        
        # 测试编码
        if "reranker" in model_path:
            # 重排序模型测试
            scores = model.compute_score([["查询文本", "相关文档"]])
            print(f"  ✅ 重排序测试成功，分数: {scores}")
        else:
            # 嵌入模型测试
            embeddings = model.encode(["测试文本"])
            print(f"  ✅ 嵌入测试成功，维度: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ {model_name} 加载失败: {e}")
        return False

def main():
    """主函数"""
    print("验证模型安装...")
    
    models = [
        ("models/bge-small-zh-v1.5", "嵌入模型"),
        ("models/bge-reranker-base", "重排序模型")
    ]
    
    success_count = 0
    for model_path, model_name in models:
        if Path(model_path).exists():
            if test_model(model_path, model_name):
                success_count += 1
        else:
            print(f"  ❌ {model_name} 目录不存在: {model_path}")
    
    print(f"\n验证结果: {success_count}/{len(models)} 模型可用")
    
    if success_count == len(models):
        print("🎉 所有模型验证成功！")
        return True
    else:
        print("❌ 部分模型验证失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

# 运行验证
python verify_models.py
```

### 3. 系统配置验证

```bash
# 使用系统配置验证
python -c "
from app.core.config import settings
import sys

print('验证系统配置...')
validation = settings.validate_paths()

for key, status in validation.items():
    status_icon = '✅' if status else '❌'
    print(f'{status_icon} {key}: {status}')

if all(validation.values()):
    print('🎉 系统配置验证成功！')
    sys.exit(0)
else:
    print('❌ 系统配置验证失败')
    sys.exit(1)
"
```

## 🔧 故障排除

### 常见问题

#### 1. 下载速度慢

**解决方案**：
```bash
# 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
python download_models.py

# 或使用代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
python download_models.py
```

#### 2. 磁盘空间不足

**解决方案**：
```bash
# 检查磁盘空间
df -h

# 清理不必要的文件
# 模型总大小约 1.5GB，确保至少有 3GB 可用空间
```

#### 3. 网络连接问题

**解决方案**：
```bash
# 测试网络连接
curl -I https://huggingface.co

# 如果无法连接，考虑使用离线下载方式
# 或联系网络管理员
```

#### 4. 权限问题

**解决方案**：
```bash
# 检查目录权限
ls -la models/

# 修复权限
chmod -R 755 models/
chown -R $USER:$USER models/
```

#### 5. 模型文件损坏

**解决方案**：
```bash
# 删除损坏的模型文件
rm -rf models/bge-small-zh-v1.5
rm -rf models/bge-reranker-base

# 重新下载
python download_models.py
```

## 📊 模型性能对比

| 模型 | 参数量 | 内存占用 | 推理速度 | 中文效果 |
|------|--------|----------|----------|----------|
| bge-small-zh-v1.5 | 33M | ~400MB | 快 | 优秀 |
| bge-base-zh-v1.5 | 102M | ~1.2GB | 中等 | 更优秀 |
| bge-large-zh-v1.5 | 326M | ~3.8GB | 慢 | 最优秀 |

**注意**：默认使用 small 版本以平衡性能和资源占用。如需更好效果，可替换为 base 或 large 版本。

## 🔄 模型更新

### 检查模型版本

```bash
# 检查当前模型版本
python -c "
import json
from pathlib import Path

for model_dir in ['models/bge-small-zh-v1.5', 'models/bge-reranker-base']:
    config_path = Path(model_dir) / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f'{model_dir}: {config.get(\"_name_or_path\", \"未知版本\")}')
"
```

### 更新模型

```bash
# 备份当前模型
mv models models_backup_$(date +%Y%m%d)

# 重新下载最新版本
python download_models.py

# 验证新模型
python verify_models.py
```

## 📝 自定义模型

如果需要使用其他模型，请修改配置文件：

```python
# app/core/config.py
class Settings(BaseSettings):
    # 修改模型目录名
    EMBEDDING_MODEL_DIR: str = "your-custom-embedding-model"
    RERANKER_MODEL_DIR: str = "your-custom-reranker-model"
```

确保自定义模型与 sentence-transformers 兼容。

---

**提示**：首次下载可能需要较长时间，请耐心等待。建议在网络状况良好时进行下载。