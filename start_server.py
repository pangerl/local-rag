#!/usr/bin/env python3
"""
Local RAG 系统启动脚本
"""

import sys
from pathlib import Path
import uvicorn

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.core.logging_config import setup_logging

# 设置日志
logger = setup_logging()

if __name__ == "__main__":
    logger.info("启动 Local RAG 系统...")
    logger.info("按 Ctrl+C 停止服务器")
    logger.info("-" * 50)

    try:
        uvicorn.run(
            "app.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=False,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("\n服务器已停止")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}", exc_info=True)
        sys.exit(1)
