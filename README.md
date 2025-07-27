# Local RAG 系统

一个轻量级、纯本地运行的检索增强生成（RAG）知识库系统。

## 项目结构

```
local-rag/
├── app/                    # 应用代码
│   ├── core/              # 核心模块
│   │   ├── config.py      # 配置管理
│   │   ├── exceptions.py  # 异常定义
│   │   └── logging_config.py # 日志配置
│   ├── services/          # 服务层
│   └── api/              # API 层
├── data/                  # 数据存储
│   └── chroma_db/        # ChromaDB 数据文件
├── models/               # 本地模型文件
│   ├── bge-small-zh-v1.5/
│   └── bge-reranker-base/
├── logs/                 # 日志文件
├── scripts/             # 批量处理脚本
├── tests/               # 测试代码
└── main.py              # 主入口文件
```

## 当前进展

✅ **任务 1: 设置项目基础结构和配置管理**
- 创建了完整的项目目录结构
- 实现了配置管理模块 (`app/core/config.py`)
- 创建了异常处理框架 (`app/core/exceptions.py`)
- 设置了日志配置系统 (`app/core/logging_config.py`)
- 配置了基础依赖和虚拟环境
- 编写了配置模块的单元测试

## 环境要求

- Python 3.13+
- uv (用于依赖管理)

## 安装和运行

1. 创建虚拟环境并安装依赖：
```bash
uv init . -p 3.13
uv add -r requirements_basic.txt
```

2. 运行基础测试：
```bash
uv run pytest tests/test_config.py -v
```

3. 测试系统启动：
```bash
uv run python main.py
```

## 配置说明

系统配置通过 `app/core/config.py` 管理，支持以下配置项：

- **模型配置**: 本地模型文件路径
- **数据库配置**: ChromaDB 存储路径
- **分片配置**: 文本分片参数
- **检索配置**: 检索和重排序参数
- **日志配置**: 日志级别和文件路径

## 下一步

接下来将实现：
- 文本处理组件（jieba 分词和滑动窗口分片）
- 模型加载器
- 向量数据库集成
- API 接口