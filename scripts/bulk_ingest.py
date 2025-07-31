#!/usr/bin/env python3
"""
批量文档导入脚本

支持单文件和目录递归处理，使用与 API 相同的 jieba 分词和滑动窗口逻辑
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import Settings
from app.services.document_service import DocumentService
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import (
    DocumentProcessError,
    UnsupportedFormatError,
    FileNotFoundError,
    DatabaseError,
    ModelLoadError
)


def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def find_documents(path: Path, recursive: bool = True) -> List[Path]:
    """
    查找指定路径下的文档文件

    Args:
        path: 文件或目录路径
        recursive: 是否递归搜索子目录

    Returns:
        List[Path]: 找到的文档文件列表
    """
    supported_extensions = {".txt", ".md", ".pdf", ".docx", ".doc"}
    documents = []

    if path.is_file():
        if path.suffix.lower() in supported_extensions:
            documents.append(path)
        else:
            logging.warning(f"跳过不支持的文件格式: {path}")
    elif path.is_dir():
        if recursive:
            # 递归搜索
            for ext in supported_extensions:
                documents.extend(path.rglob(f"*{ext}"))
        else:
            # 只搜索当前目录
            for ext in supported_extensions:
                documents.extend(path.glob(f"*{ext}"))
    else:
        logging.error(f"路径不存在: {path}")

    return sorted(documents)


def process_documents(document_service: DocumentService, documents: List[Path],
                     chunk_size: int, chunk_overlap: int) -> Dict[str, Any]:
    """
    批量处理文档

    Args:
        document_service: 文档处理服务
        documents: 文档文件列表
        chunk_size: 分片大小
        chunk_overlap: 分片重叠

    Returns:
        Dict[str, Any]: 处理结果统计
    """
    start_time = time.time()

    results = {
        "total_documents": len(documents),
        "successful_documents": 0,
        "failed_documents": 0,
        "total_chunks_created": 0,
        "total_chunks_stored": 0,
        "processing_results": [],
        "errors": []
    }

    logging.info(f"开始批量处理 {len(documents)} 个文档")

    for i, doc_path in enumerate(documents, 1):
        try:
            logging.info(f"[{i}/{len(documents)}] 处理文档: {doc_path}")

            # 处理单个文档
            result = document_service.process_document(
                document_path=str(doc_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            results["processing_results"].append(result)
            results["successful_documents"] += 1
            results["total_chunks_created"] += result["chunks_created"]
            results["total_chunks_stored"] += result["chunks_stored"]

            logging.info(f"文档处理成功: {doc_path}, "
                        f"分片数: {result['chunks_created']}, "
                        f"耗时: {result['processing_time']:.2f}s")

        except (FileNotFoundError, UnsupportedFormatError, DocumentProcessError) as e:
            error_info = {
                "document_path": str(doc_path),
                "error": str(e),
                "error_type": type(e).__name__
            }
            results["errors"].append(error_info)
            results["failed_documents"] += 1

            logging.error(f"文档处理失败: {doc_path}, 错误: {str(e)}")

        except (DatabaseError, ModelLoadError) as e:
            # 这些是系统级错误，应该停止处理
            logging.error(f"系统错误，停止处理: {str(e)}")
            error_info = {
                "document_path": str(doc_path),
                "error": str(e),
                "error_type": type(e).__name__
            }
            results["errors"].append(error_info)
            results["failed_documents"] += 1
            break

        except Exception as e:
            error_info = {
                "document_path": str(doc_path),
                "error": str(e),
                "error_type": type(e).__name__
            }
            results["errors"].append(error_info)
            results["failed_documents"] += 1

            logging.error(f"未知错误: {doc_path}, 错误: {str(e)}")

    total_time = time.time() - start_time
    results["total_processing_time"] = total_time

    logging.info(f"批量处理完成:")
    logging.info(f"  总文档数: {results['total_documents']}")
    logging.info(f"  成功处理: {results['successful_documents']}")
    logging.info(f"  处理失败: {results['failed_documents']}")
    logging.info(f"  总分片数: {results['total_chunks_created']}")
    logging.info(f"  总耗时: {total_time:.2f}s")

    if results["errors"]:
        logging.info("失败的文档:")
        for error in results["errors"]:
            logging.info(f"  {error['document_path']}: {error['error']}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量文档导入工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单个文件
  python scripts/bulk_ingest.py --path documents/example.txt

  # 处理目录（递归）
  python scripts/bulk_ingest.py --path documents/ --chunk-size 500 --chunk-overlap 100

  # 处理目录（非递归）
  python scripts/bulk_ingest.py --path documents/ --no-recursive

  # 使用自定义参数
  python scripts/bulk_ingest.py --path documents/ --chunk-size 200 --chunk-overlap 30 --log-level DEBUG
        """
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="文档文件或目录路径"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="文本分片大小（词元数量），默认: 500"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="相邻分片重叠（词元数量），默认: 50"
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="处理目录时不递归搜索子目录"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别，默认: INFO"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要处理的文件，不实际处理"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    # 验证参数
    if args.chunk_overlap >= args.chunk_size:
        logging.error("chunk-overlap 必须小于 chunk-size")
        sys.exit(1)

    # 验证路径
    path = Path(args.path)
    if not path.exists():
        logging.error(f"路径不存在: {path}")
        sys.exit(1)

    try:
        # 查找文档文件
        logging.info(f"搜索文档文件: {path}")
        documents = find_documents(path, recursive=not args.no_recursive)

        if not documents:
            logging.warning("未找到支持的文档文件")
            sys.exit(0)

        logging.info(f"找到 {len(documents)} 个文档文件:")
        for doc in documents:
            logging.info(f"  {doc}")

        # 如果是 dry-run，只显示文件列表
        if args.dry_run:
            logging.info("Dry-run 模式，不执行实际处理")
            sys.exit(0)

        # 初始化服务
        logging.info("初始化系统服务...")
        settings = Settings()
        db_service = ChromaDBService(settings)
        model_loader = ModelLoader(settings)
        document_service = DocumentService(settings, db_service, model_loader)

        # 检查系统健康状态
        health = document_service.health_check()
        if health["status"] != "healthy":
            logging.error(f"系统健康检查失败: {health}")
            sys.exit(1)

        logging.info("系统服务初始化完成")

        # 批量处理文档
        results = process_documents(
            document_service=document_service,
            documents=documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        # 输出最终统计
        success_rate = (results["successful_documents"] / results["total_documents"]) * 100
        logging.info(f"处理完成，成功率: {success_rate:.1f}%")

        # 根据结果设置退出码
        if results["failed_documents"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        logging.info("用户中断处理")
        sys.exit(130)

    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()