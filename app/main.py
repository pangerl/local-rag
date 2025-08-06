"""
FastAPI 应用程序主入口
配置 FastAPI 应用、中间件和路由
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.core.config import Settings
from app.core.logging_config import setup_logging
from app.core.exceptions import LocalRAGException, ModelLoadError, DatabaseError
from app.api.routes import router, admin_router, model_manager
from app.middleware.exception_handler import exception_handler

# 全局配置
settings = Settings()

# 设置日志系统
logger = setup_logging()

def _validate_environment():
    """验证运行环境"""
    logger.info("验证运行环境...")
    if sys.version_info < (3, 8):
        raise RuntimeError(f"需要 Python 3.8+，当前版本: {sys.version}")
    try:
        settings.chroma_db_full_path.mkdir(parents=True, exist_ok=True)
        settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("必要目录创建完成")
    except Exception as e:
        logger.error(f"创建目录失败: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时初始化
    logger.info("=" * 50)
    logger.info("Local RAG 系统启动中...")
    logger.info("=" * 50)

    try:
        # 验证环境
        _validate_environment()

        # 记录配置信息
        logger.info(f"API 服务地址: http://{settings.API_HOST}:{settings.API_PORT}")
        logger.info(f"嵌入模型路径: {settings.embedding_model_path}")
        logger.info(f"重排序模型路径: {settings.reranker_model_path}")
        logger.info(f"ChromaDB 路径: {settings.chroma_db_full_path}")
        logger.info(f"日志级别: {settings.LOG_LEVEL}")

        # 验证关键路径
        path_validation = settings.validate_paths()
        logger.info(f"路径验证结果: {path_validation}")
        if not all(path_validation.values()):
            logger.error("关键路径验证失败，请检查配置和文件是否存在。")
            # 可以在这里决定是否要引发异常并停止启动
            # raise RuntimeError("关键路径验证失败")

        # 加载模型和服务
        logger.info("加载模型和服务...")
        model_manager.load_models()


        logger.info("=" * 50)
        logger.info("Local RAG 系统启动完成！")
        logger.info("API 文档地址: http://{}:{}/docs".format(settings.API_HOST, settings.API_PORT))
        logger.info("=" * 50)

        yield

    except ModelLoadError as e:
        logger.error(f"模型加载失败: {str(e)}")
        logger.error("请检查模型文件是否存在且完整")
        raise

    except DatabaseError as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        logger.error("请检查数据库配置和权限")
        raise

    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}", exc_info=True)
        raise

    finally:
        # 关闭时清理
        logger.info("=" * 50)
        logger.info("Local RAG 系统关闭中...")
        logger.info("=" * 50)
        try:
            logger.info("执行清理操作...")
            model_manager.unload_models()
            logger.info("清理操作完成")
        except Exception as e:
            logger.error(f"清理过程中发生错误: {str(e)}", exc_info=True)
        logger.info("Local RAG 系统已关闭")
        logger.info("=" * 50)


# 创建 FastAPI 应用
app = FastAPI(
    title="Local RAG API",
    description="""
    Local RAG (Retrieval-Augmented Generation) 知识库系统

    ## 功能特性

    * **文档摄取**: 支持 .txt 和 .md 格式文档的上传和处理
    * **智能分片**: 基于词元数量的滑动窗口文本分片
    * **向量检索**: 使用嵌入模型进行相似性搜索
    * **结果重排序**: 使用重排序模型提高检索精度
    * **完全离线**: 所有模型和数据均在本地运行

    ## 使用说明

    1. 首先使用 `/api/v1/ingest` 接口上传和处理文档
    2. 然后使用 `/api/v1/retrieve` 接口检索相关信息
    3. 使用 `/api/v1/health` 检查系统状态
    """,
    version="1.1.0",
    contact={
        "name": "Local RAG System",
        "url": "https://github.com/pangerl/local-rag",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册统一异常处理器
app.add_exception_handler(ValidationError, exception_handler.handle_validation_error)
app.add_exception_handler(LocalRAGException, exception_handler.handle_local_rag_exception)
app.add_exception_handler(HTTPException, exception_handler.handle_http_exception)
app.add_exception_handler(Exception, exception_handler.handle_general_exception)


# 注册路由
app.include_router(router)
app.include_router(admin_router)


@app.get("/", tags=["Root"])
async def root():
    """根路径，返回 API 基本信息"""
    return {
        "name": "Local RAG API",
        "version": "1.0.0",
        "description": "本地运行的检索增强生成知识库系统",
        "status": "running",
        "docs_url": "/docs",
        "admin_page": "/admin",
        "health_check": "/api/v1/health",
        "endpoints": {
            "admin": "/admin",
            "ingest": "/api/v1/ingest",
            "retrieve": "/api/v1/retrieve",
            "health": "/api/v1/health"
        }
    }


@app.get("/health", tags=["Health"])
async def simple_health_check():
    """简单健康检查，不依赖服务初始化"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }
