# Requirements Document

## Introduction

Local-RAG 是一个轻量级、纯本地运行的检索增强生成（RAG）知识库系统。该系统专为向外部系统提供服务而设计，支持通过 API 接口进行实时文档处理，以及通过命令行脚本进行大规模数据初始化。核心功能包括文档上传、处理、向量化存储，并根据用户查询高效检索和重排序相关信息片段。系统采用中文优化的文本分片策略，使用 jieba 分词结合滑动窗口技术，确保完全离线运行以最大化数据隐私与安全。

## Requirements

### Requirement 1

**User Story:** 作为系统管理员，我希望能够完全离线部署和运行 RAG 系统，以确保数据隐私和安全性。

#### Acceptance Criteria

1. WHEN 系统启动时 THEN 系统 SHALL 从本地文件系统加载所有必需的模型文件
2. WHEN 系统运行时 THEN 系统 SHALL NOT 发起任何网络请求来下载模型或数据
3. WHEN 配置模型路径时 THEN 系统 SHALL 使用 models/bge-small-zh-v1.5/ 作为嵌入模型路径
4. WHEN 配置模型路径时 THEN 系统 SHALL 使用 models/bge-reranker-base/ 作为重排序模型路径

### Requirement 2

**User Story:** 作为开发者，我希望通过 FastAPI 接口上传和处理文档，以便实时集成到外部系统中。

#### Acceptance Criteria

1. WHEN 调用 /api/v1/ingest 接口时 THEN 系统 SHALL 接受包含 document_path、chunk_size、chunk_overlap 的 JSON 请求
2. WHEN 处理文档时 THEN 系统 SHALL 使用 jieba 分词对中文文本进行分词处理
3. WHEN 分片文档时 THEN 系统 SHALL 基于词元数量而非字符数量进行滑动窗口分片
4. WHEN chunk_size 设置为 300 时 THEN 系统 SHALL 创建包含 300 个词元的文本分片
5. WHEN chunk_overlap 设置为 50 时 THEN 系统 SHALL 在相邻分片间保持 50 个词元的重叠
6. WHEN 文档处理完成时 THEN 系统 SHALL 将分片向量化并存储到 ChromaDB 中

### Requirement 3

**User Story:** 作为用户，我希望能够查询知识库并获得相关的信息片段，以便快速找到所需信息。

#### Acceptance Criteria

1. WHEN 调用 /api/v1/retrieve 接口时 THEN 系统 SHALL 接受包含查询文本的请求
2. WHEN 执行检索时 THEN 系统 SHALL 默认检索 10 个候选文档片段 (retrieval_k=10)
3. WHEN 执行重排序时 THEN 系统 SHALL 使用 bge-reranker-base 模型对候选片段进行重排序
4. WHEN 返回结果时 THEN 系统 SHALL 默认返回前 3 个最相关的文档片段 (top_k=3)
5. WHEN 返回结果时 THEN 系统 SHALL 包含文档片段内容和相关性分数

### Requirement 4

**User Story:** 作为数据管理员，我希望能够批量导入大量文档，以便高效地初始化知识库。

#### Acceptance Criteria

1. WHEN 运行 bulk_ingest.py 脚本时 THEN 系统 SHALL 接受 --path 参数指定文档或目录路径
2. WHEN 指定 --chunk-size 参数时 THEN 系统 SHALL 使用指定的词元数量进行分片
3. WHEN 指定 --chunk-overlap 参数时 THEN 系统 SHALL 使用指定的词元重叠数量
4. WHEN 未指定参数时 THEN 系统 SHALL 使用默认值 chunk_size=300, chunk_overlap=50
5. WHEN 处理目录时 THEN 系统 SHALL 递归处理目录下的所有文档文件
6. WHEN 批量导入时 THEN 系统 SHALL 使用与 API 接口相同的 jieba 分词和滑动窗口逻辑

### Requirement 5

**User Story:** 作为开发者，我希望系统具有良好的项目结构和代码质量，以便于维护和扩展。

#### Acceptance Criteria

1. WHEN 组织项目结构时 THEN 系统 SHALL 使用 app/、data/、models/、scripts/、tests/ 的目录结构
2. WHEN 管理依赖时 THEN 系统 SHALL 使用 uv 作为环境和依赖管理工具
3. WHEN 编写代码时 THEN 系统 SHALL 使用 Python 3.13+ 版本
4. WHEN 提供 Web 服务时 THEN 系统 SHALL 使用 FastAPI 框架
5. WHEN 存储向量数据时 THEN 系统 SHALL 使用 ChromaDB 作为本地向量数据库
6. WHEN 加载模型时 THEN 系统 SHALL 使用 sentence-transformers 框架

### Requirement 6

**User Story:** 作为系统用户，我希望系统针对中文文本进行优化，以获得更好的处理效果。

#### Acceptance Criteria

1. WHEN 处理中文文本时 THEN 系统 SHALL 使用 jieba 库进行中文分词
2. WHEN 创建文本分片时 THEN 系统 SHALL 确保分片在语义上连贯且大小可控
3. WHEN 使用嵌入模型时 THEN 系统 SHALL 使用 BAAI/bge-small-zh-v1.5 中文优化模型
4. WHEN 使用重排序模型时 THEN 系统 SHALL 使用 BAAI/bge-reranker-base 模型
5. WHEN 重组词元为文本时 THEN 系统 SHALL 正确连接词元以保持中文文本的可读性

### Requirement 7

**User Story:** 作为系统管理员，我希望能够轻松配置和部署系统，以便快速启动服务。

#### Acceptance Criteria

1. WHEN 配置系统时 THEN 系统 SHALL 通过 app/core/config.py 集中管理配置项
2. WHEN 启动 API 服务时 THEN 系统 SHALL 支持通过 uvicorn 在指定端口启动服务
3. WHEN 准备模型文件时 THEN 系统 SHALL 提供清晰的模型下载和部署指南
4. WHEN 安装依赖时 THEN 系统 SHALL 通过 requirements.txt 管理所有必需的 Python 包
5. WHEN 运行系统时 THEN 系统 SHALL 在 data/chroma_db/ 目录下存储向量数据库文件

### Requirement 8

**User Story:** 作为开发者，我希望系统能够处理各种错误情况并提供清晰的错误信息，以便快速定位和解决问题。

#### Acceptance Criteria

1. WHEN 模型文件不存在或损坏时 THEN 系统 SHALL 返回明确的错误信息并停止启动
2. WHEN 文档路径无效或文件无法读取时 THEN 系统 SHALL 返回 HTTP 400 错误和具体错误描述
3. WHEN ChromaDB 连接失败时 THEN 系统 SHALL 返回 HTTP 500 错误和数据库连接错误信息
4. WHEN API 请求参数验证失败时 THEN 系统 SHALL 返回 HTTP 422 错误和参数验证详情
5. WHEN 文档处理过程中发生异常时 THEN 系统 SHALL 记录错误日志并返回处理失败信息

### Requirement 9

**User Story:** 作为系统用户，我希望系统明确支持的文档格式，以便正确准备输入文档。

#### Acceptance Criteria

1. WHEN 处理文档时 THEN 系统 SHALL 仅支持 .txt 和 .md 格式的文本文件
2. WHEN 上传非支持格式文件时 THEN 系统 SHALL 返回格式不支持的错误信息
3. WHEN 检测文件格式时 THEN 系统 SHALL 基于文件扩展名进行格式判断
4. WHEN 读取 .txt 文件时 THEN 系统 SHALL 使用 UTF-8 编码读取文件内容
5. WHEN 读取 .md 文件时 THEN 系统 SHALL 提取纯文本内容忽略 Markdown 格式标记

### Requirement 10

**User Story:** 作为系统管理员，我希望系统提供基本的监控和日志功能，以便了解系统运行状态和排查问题。

#### Acceptance Criteria

1. WHEN 系统运行时 THEN 系统 SHALL 记录所有 API 请求和响应的基本信息到日志文件
2. WHEN 处理文档时 THEN 系统 SHALL 记录文档处理的开始、完成和耗时信息
3. WHEN 发生错误时 THEN 系统 SHALL 记录详细的错误堆栈信息到错误日志
4. WHEN 系统启动时 THEN 系统 SHALL 记录模型加载状态和配置信息
5. WHEN 查询知识库时 THEN 系统 SHALL 记录查询内容、检索结果数量和响应时间