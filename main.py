"""
Local RAG 系统主入口
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.exceptions import LocalRAGException


def main():
    """主函数"""
    try:
        # 初始化日志系统
        logger = setup_logging()
        logger.info("Local RAG 系统启动中...")
        
        # 验证配置
        logger.info(f"模型基础路径: {settings.MODEL_BASE_PATH}")
        logger.info(f"嵌入模型路径: {settings.embedding_model_path}")
        logger.info(f"重排序模型路径: {settings.reranker_model_path}")
        logger.info(f"ChromaDB 路径: {settings.chroma_db_full_path}")
        
        # 验证关键路径
        validation_results = settings.validate_paths()
        logger.info(f"路径验证结果: {validation_results}")
        
        # 检查必要目录是否存在，不存在则创建
        settings.chroma_db_full_path.mkdir(parents=True, exist_ok=True)
        settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Local RAG 系统基础配置完成")
        
    except Exception as e:
        print(f"系统启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()