import sys
import os
import logging

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import settings
from app.services.model_loader import ModelLoader
from app.core.exceptions import ModelLoadError

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_reranker():
    """
    调试重排序模型的加载和预测功能
    """
    logger.info("开始调试重排序模型...")

    try:
        # 1. 初始化模型加载器
        logger.info("初始化 ModelLoader...")
        model_loader = ModelLoader(settings)
        logger.info(f"重排序模型路径: {settings.reranker_model_path}")

        # 2. 验证模型文件
        logger.info("验证模型文件...")
        model_loader.validate_model_files()
        logger.info("模型文件验证通过。")

        # 3. 加载重排序模型
        logger.info("加载重排序模型...")
        reranker_model = model_loader.load_reranker_model()
        logger.info("重排序模型加载成功。")

        # 4. 准备测试数据
        test_query = "什么是 Python？"
        test_docs = [
            "Python 是一种解释型、高级、通用编程语言。",
            "Java 是一种广泛使用的计算机编程语言。",
            "C++ 是一种通用的、编译型的编程语言。"
        ]
        query_doc_pairs = [[test_query, doc] for doc in test_docs]
        logger.info(f"准备测试数据: {len(query_doc_pairs)} 个查询-文档对。")

        # 5. 执行预测
        logger.info("执行重排序预测...")
        scores = reranker_model.predict(query_doc_pairs, show_progress_bar=True)
        logger.info("预测完成。")

        # 6. 打印结果
        logger.info("--- 调试结果 ---")
        for i, score in enumerate(scores):
            logger.info(f"  查询: '{test_query}'")
            logger.info(f"  文档: '{test_docs[i]}'")
            logger.info(f"  分数: {score:.4f}")
            logger.info("-" * 20)

        logger.info("重排序模型调试成功！")

    except ModelLoadError as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"调试过程中发生未知错误: {e}", exc_info=True)

if __name__ == "__main__":
    debug_reranker()
