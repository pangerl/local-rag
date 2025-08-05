"""
文档处理服务模块
整合文本分片、向量化和存储的完整文档处理流程
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps

from app.core.config import Settings
from app.core.document_processor import DocumentProcessor
from app.core.chunker import TextChunker
from app.services.vector_store import VectorStore
from app.services.database import ChromaDBService
from app.services.models import EmbeddingModel
from app.core.exceptions import (
    DocumentProcessError,
    UnsupportedFormatError,
    FileNotFoundError,
    DatabaseError,
    ModelLoadError
)


logger = logging.getLogger(__name__)


def _handle_service_errors(method_name: str):
    """
    装饰器工厂，用于统一处理服务层中的异常。
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # 动态获取 document_path 用于日志
            log_target = "N/A"
            if 'document_path' in kwargs:
                log_target = kwargs['document_path']
            elif 'document_paths' in kwargs:
                log_target = f"{len(kwargs['document_paths'])} documents"
            elif 'directory_path' in kwargs:
                log_target = kwargs['directory_path']
            elif args:
                log_target = args[0]

            start_time = time.time()
            log_suffix = f": {log_target}" if log_target != "N/A" else ""
            logger.info(f"开始 {method_name}{log_suffix}")

            try:
                result = method(self, *args, **kwargs)
                processing_time = time.time() - start_time
                logger.info(f"{method_name} 完成{log_suffix}, 耗时: {processing_time:.2f}s")
                return result
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = f"{method_name} 失败{log_suffix}, 耗时: {processing_time:.2f}s, 错误: {e}"
                logger.error(error_msg)
                if isinstance(e, (FileNotFoundError, UnsupportedFormatError, DocumentProcessError, DatabaseError, ModelLoadError)):
                    raise
                else:
                    raise DocumentProcessError(error_msg) from e
        return wrapper
    return decorator


class DocumentService:
    """
    文档处理服务类

    整合文本分片、向量化和存储的完整文档处理流程，
    提供文档处理过程的错误处理、异常管理和性能监控
    """

    def __init__(self, settings: Settings, db_service: ChromaDBService, embedding_model: EmbeddingModel):
        """
        初始化文档处理服务

        Args:
            settings: 系统配置对象
            db_service: ChromaDB 数据库服务
            embedding_model: 嵌入模型实例
        """
        self.settings = settings
        self.db_service = db_service
        self.embedding_model = embedding_model

        # 初始化组件
        self.document_processor = DocumentProcessor()
        self.text_splitter = TextChunker()
        self.vector_store = VectorStore(settings, db_service, embedding_model)

        # 性能监控统计
        self.processing_stats = {
            "total_documents_processed": 0,
            "total_chunks_created": 0,
            "total_processing_time": 0.0,
            "last_processed_at": None
        }

        logger.info("文档处理服务初始化完成")

    @_handle_service_errors("处理文档")
    def process_document(self, document_path: str, original_filename: str, file_size: int,
                         chunk_size: Optional[int] = None,
                         chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
        """
        处理单个文档的完整流程

        Args:
            document_path: 文档在磁盘上的临时路径
            original_filename: 用户上传的原始文件名
            file_size: 文件大小（字节）
            chunk_size: 分片大小（词元数量），如果为 None 则使用默认值
            chunk_overlap: 分片重叠（词元数量），如果为 None 则使用默认值

        Returns:
            Dict[str, Any]: 处理结果信息
        """
        start_time = time.time()  # 保留用于精确的 _update_processing_stats

        # 1. Load: 加载文档为 Document 对象
        documents = self.document_processor.load(document_path)
        if not documents:
            logger.warning(f"文档加载后为空，处理终止: {document_path}")
            return {"status": "skipped", "message": "Document is empty or unreadable."}

        # 提前计算 text_length
        text_length = sum(len(doc.page_content) for doc in documents)

        # 确定 chunk_size
        final_chunk_size = chunk_size or self.settings.DEFAULT_CHUNK_SIZE
        final_chunk_overlap = chunk_overlap or self.settings.DEFAULT_CHUNK_OVERLAP

        # 2. Split: 将 Document 分割成块
        split_docs = self.text_splitter.split_documents(
            documents,
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap
        )

        # 3. Store: 将分割后的文档存入向量数据库
        storage_result = self.vector_store.add_documents(
            split_docs,
            document_path=original_filename,
            file_size=file_size,
            text_length=text_length,
            chunk_size=final_chunk_size
        )

        # 4. 更新统计信息
        processing_time = time.time() - start_time
        self._update_processing_stats(len(split_docs), processing_time)

        # 5. 准备返回结果
        return {
            "document_path": original_filename,
            "status": "success",
            "text_length": text_length,
            "chunks_created": len(split_docs),
            "chunks_stored": storage_result.get("chunks_stored", 0),
            "processing_time": processing_time,
            "chunk_size": final_chunk_size,
            "chunk_overlap": final_chunk_overlap,
            "collection_name": storage_result.get("collection_name")
        }

    @_handle_service_errors("删除文档")
    def delete_document(self, document_path: str) -> Dict[str, Any]:
        """
        删除已存储的文档

        Args:
            document_path: 文档路径

        Returns:
            Dict[str, Any]: 删除结果信息
        """
        start_time = time.time()
        delete_result = self.vector_store.delete_document_chunks(document_path)
        processing_time = time.time() - start_time

        return {
            "document_path": document_path,
            "status": delete_result["status"],
            "chunks_deleted": delete_result.get("chunks_deleted", 0),
            "processing_time": processing_time
        }

    @_handle_service_errors("获取文档信息")
    def get_document_info(self, document_path: str) -> Dict[str, Any]:
        """
        获取文档信息

        Args:
            document_path: 文档路径

        Returns:
            Dict[str, Any]: 文档信息
        """
        file_info = self.document_processor.get_file_info(document_path)
        chunks_info = self.vector_store.get_document_chunks(document_path)

        return {
            "document_path": document_path,
            "file_info": file_info,
            "storage_info": {
                "is_stored": chunks_info["status"] == "success",
                "total_chunks": chunks_info.get("total_chunks", 0),
                "chunks": chunks_info.get("chunks", [])
            }
        }

    def batch_process_documents(self, document_paths: List[str],
                               chunk_size: Optional[int] = None,
                               chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
        """
        批量处理多个文档

        Args:
            document_paths: 文档路径列表
            chunk_size: 分片大小（词元数量）
            chunk_overlap: 分片重叠（词元数量）

        Returns:
            Dict[str, Any]: 批量处理结果
        """
        start_time = time.time()
        logger.info(f"开始批量处理 {len(document_paths)} 个文档")

        results = {
            "total_documents": len(document_paths),
            "successful_documents": 0,
            "failed_documents": 0,
            "processing_results": [],
            "errors": []
        }

        for path_str in document_paths:
            try:
                p = Path(path_str)
                if not p.is_file():
                    raise FileNotFoundError(f"文件不存在: {path_str}")

                file_size = p.stat().st_size
                # 使用文件路径作为 original_filename，因为这是批量处理的上下文
                original_filename = path_str

                result = self.process_document(
                    document_path=path_str,
                    original_filename=original_filename,
                    file_size=file_size,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                results["processing_results"].append(result)
                results["successful_documents"] += 1

            except Exception as e:
                error_info = {
                    "document_path": path_str,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                results["errors"].append(error_info)
                results["failed_documents"] += 1
                logger.error(f"批量处理中文档失败: {path_str}, 错误: {e}")

        total_time = time.time() - start_time
        results["total_processing_time"] = total_time

        logger.info(f"批量处理完成，成功: {results['successful_documents']}, "
                   f"失败: {results['failed_documents']}, "
                   f"总耗时: {total_time:.2f}s")

        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息

        Returns:
            Dict[str, Any]: 处理统计信息
        """
        stats = self.processing_stats.copy()

        # 添加存储统计信息
        try:
            storage_stats = self.vector_store.get_storage_stats()
            stats.update({
                "storage_total_documents": storage_stats.get("total_documents", 0),
                "storage_total_chunks": storage_stats.get("total_chunks", 0),
                "average_chunks_per_document": storage_stats.get("average_chunks_per_document", 0)
            })

            # 添加详细的文档列表信息
            documents_list = self.db_service.list_documents()
            stats["documents"] = documents_list

        except Exception as e:
            logger.warning(f"获取存储统计信息失败: {str(e)}")

        return stats

    @_handle_service_errors("处理目录")
    def process_directory(self, directory_path: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
        """
        批量处理目录中的所有支持的文档

        Args:
            directory_path: 包含文档的目录路径
            chunk_size: 分片大小（词元数量）
            chunk_overlap: 分片重叠（词元数量）

        Returns:
            Dict[str, Any]: 批量处理结果
        """
        supported_extensions = self.document_processor.SUPPORTED_FORMATS
        files_to_process = [
            p for p in Path(directory_path).rglob('*')
            if p.is_file() and p.suffix.lower() in supported_extensions
        ]

        paths_to_process = [str(p) for p in files_to_process]

        return self.batch_process_documents(
            document_paths=paths_to_process,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @_handle_service_errors("列出带统计的文档")
    def list_documents_with_stats(self) -> List[Dict[str, Any]]:
        """
        获取包含统计信息的文档列表

        Returns:
            List[Dict[str, Any]]: 包含每个文档详细信息的列表
        """
        return self.db_service.list_documents()

    @_handle_service_errors("获取系统统计")
    def get_system_stats(self) -> Dict[str, Any]:
        """
        获取系统层面的统计信息

        Returns:
            Dict[str, Any]: 系统统计信息，如文档总数和分片总数
        """
        storage_stats = self.vector_store.get_storage_stats()
        return {
            "total_documents": storage_stats.get("total_documents", 0),
            "total_chunks": storage_stats.get("total_chunks", 0),
        }

    def _update_processing_stats(self, chunks_count: int, processing_time: float):
        """
        更新处理统计信息

        Args:
            chunks_count: 分片数量
            processing_time: 处理时间
        """
        self.processing_stats["total_documents_processed"] += 1
        self.processing_stats["total_chunks_created"] += chunks_count
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["last_processed_at"] = datetime.now().isoformat()

        logger.debug(f"统计信息更新: 总文档数: {self.processing_stats['total_documents_processed']}, "
                    f"总分片数: {self.processing_stats['total_chunks_created']}")

    def health_check(self) -> Dict[str, Any]:
        """
        服务健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        health_info = {
            "status": "unknown",
            "components": {},
            "error": None
        }

        try:
            # 检查数据库服务
            db_health = self.db_service.health_check()
            health_info["components"]["database"] = db_health

            # 检查文档处理器
            health_info["components"]["document_processor"] = {
                "status": "healthy",
                "supported_formats": list(self.document_processor.SUPPORTED_FORMATS)
            }

            # 检查分词器
            health_info["components"]["text_splitter"] = {
                "status": "healthy",
                "type": self.text_splitter.__class__.__name__
            }

            # 确定整体状态
            health_info["status"] = "healthy" if db_health["status"] == "healthy" else "unhealthy"

        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
            logger.error(f"服务健康检查失败: {str(e)}")

        return health_info
