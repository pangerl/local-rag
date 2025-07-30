"""
文档处理服务模块
整合文本分片、向量化和存储的完整文档处理流程
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.config import Settings
from app.core.document_processor import DocumentProcessor
from app.core.chunker import TextChunker
from app.services.vector_store import VectorStore
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import (
    DocumentProcessError,
    UnsupportedFormatError,
    FileNotFoundError,
    DatabaseError,
    ModelLoadError
)


logger = logging.getLogger(__name__)


class DocumentService:
    """
    文档处理服务类

    整合文本分片、向量化和存储的完整文档处理流程，
    提供文档处理过程的错误处理、异常管理和性能监控
    """

    def __init__(self, settings: Settings, db_service: ChromaDBService,
                 model_loader: ModelLoader):
        """
        初始化文档处理服务

        Args:
            settings: 系统配置对象
            db_service: ChromaDB 数据库服务
            model_loader: 模型加载器
        """
        self.settings = settings
        self.db_service = db_service
        self.model_loader = model_loader

        # 初始化组件
        self.document_processor = DocumentProcessor()
        self.text_splitter = TextChunker()
        self.vector_store = VectorStore(settings, db_service, model_loader)

        # 性能监控统计
        self.processing_stats = {
            "total_documents_processed": 0,
            "total_chunks_created": 0,
            "total_processing_time": 0.0,
            "last_processed_at": None
        }

        logger.info("文档处理服务初始化完成")

    def process_document(self, document_path: str, chunk_size: Optional[int] = None,
                        chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
        """
        处理单个文档的完整流程

        Args:
            document_path: 文档路径
            chunk_size: 分片大小（词元数量），如果为 None 则使用默认值
            chunk_overlap: 分片重叠（词元数量），如果为 None 则使用默认值

        Returns:
            Dict[str, Any]: 处理结果信息

        Raises:
            FileNotFoundError: 当文件不存在时抛出异常
            UnsupportedFormatError: 当文件格式不支持时抛出异常
            DocumentProcessError: 当文档处理失败时抛出异常
        """
        start_time = time.time()

        try:
            logger.info(f"开始处理文档 (LangChain 流程): {document_path}")

            # 1. Load: 加载文档为 Document 对象
            documents = self.document_processor.load(document_path)
            if not documents:
                logger.warning(f"文档加载后为空，处理终止: {document_path}")
                return {"status": "skipped", "message": "Document is empty or unreadable."}

            # 2. Split: 将 Document 分割成块
            split_docs = self.text_splitter.split_documents(
                documents,
                chunk_size=chunk_size or self.settings.DEFAULT_CHUNK_SIZE,
                chunk_overlap=chunk_overlap or self.settings.DEFAULT_CHUNK_OVERLAP
            )

            # 3. Store: 将分割后的文档存入向量数据库
            storage_result = self.vector_store.add_documents(split_docs)

            # 4. 更新统计信息
            processing_time = time.time() - start_time
            self._update_processing_stats(len(split_docs), processing_time)

            # 5. 准备返回结果
            text_length = sum(len(doc.page_content) for doc in documents)
            result = {
                "document_path": document_path,
                "status": "success",
                "text_length": text_length,
                "chunks_created": len(split_docs),
                "chunks_stored": storage_result.get("chunks_stored", 0),
                "processing_time": processing_time,
                "chunk_size": chunk_size or self.settings.DEFAULT_CHUNK_SIZE,
                "chunk_overlap": chunk_overlap or self.settings.DEFAULT_CHUNK_OVERLAP,
                "collection_name": storage_result.get("collection_name")
            }

            logger.info(f"文档处理完成: {document_path}, "
                       f"耗时: {processing_time:.2f}s, "
                       f"分片数: {len(split_docs)}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"文档处理失败: {document_path}, 耗时: {processing_time:.2f}s, 错误: {str(e)}"
            logger.error(error_msg)

            if isinstance(e, (FileNotFoundError, UnsupportedFormatError,
                            DocumentProcessError, DatabaseError, ModelLoadError)):
                raise
            else:
                raise DocumentProcessError(error_msg) from e

    def update_document(self, document_path: str, chunk_size: Optional[int] = None,
                       chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
        """
        更新已存储的文档

        Args:
            document_path: 文档路径
            chunk_size: 分片大小（词元数量）
            chunk_overlap: 分片重叠（词元数量）

        Returns:
            Dict[str, Any]: 更新结果信息
        """
        start_time = time.time()

        try:
            logger.info(f"开始更新文档 (LangChain 流程): {document_path}")

            # 1. 先删除旧的文档分片
            delete_result = self.vector_store.delete_document_chunks(document_path)

            # 2. 使用标准处理流程添加新文档
            # 注意：这里复用了 process_document 的逻辑
            add_result = self.process_document(document_path, chunk_size, chunk_overlap)

            # 3. 组合结果
            processing_time = time.time() - start_time
            result = {
                "document_path": document_path,
                "status": "success",
                "text_length": add_result.get("text_length", 0),
                "old_chunks_deleted": delete_result.get("chunks_deleted", 0),
                "new_chunks_stored": add_result.get("chunks_stored", 0),
                "processing_time": processing_time,
                "chunk_size": add_result.get("chunk_size"),
                "chunk_overlap": add_result.get("chunk_overlap")
            }

            logger.info(f"文档更新完成: {document_path}, "
                       f"耗时: {processing_time:.2f}s, "
                       f"删除旧分片: {result['old_chunks_deleted']}, "
                       f"添加新分片: {result['new_chunks_stored']}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"文档更新失败: {document_path}, 耗时: {processing_time:.2f}s, 错误: {str(e)}"
            logger.error(error_msg)

            if isinstance(e, (FileNotFoundError, UnsupportedFormatError,
                            DocumentProcessError, DatabaseError, ModelLoadError)):
                raise
            else:
                raise DocumentProcessError(error_msg) from e

    def delete_document(self, document_path: str) -> Dict[str, Any]:
        """
        删除已存储的文档

        Args:
            document_path: 文档路径

        Returns:
            Dict[str, Any]: 删除结果信息
        """
        start_time = time.time()

        try:
            logger.info(f"开始删除文档: {document_path}")

            # 删除向量存储中的文档分片
            delete_result = self.vector_store.delete_document_chunks(document_path)

            processing_time = time.time() - start_time

            result = {
                "document_path": document_path,
                "status": delete_result["status"],
                "chunks_deleted": delete_result.get("chunks_deleted", 0),
                "processing_time": processing_time
            }

            logger.info(f"文档删除完成: {document_path}, "
                       f"耗时: {processing_time:.2f}s, "
                       f"删除分片数: {result['chunks_deleted']}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"文档删除失败: {document_path}, 耗时: {processing_time:.2f}s, 错误: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessError(error_msg) from e

    def get_document_info(self, document_path: str) -> Dict[str, Any]:
        """
        获取文档信息

        Args:
            document_path: 文档路径

        Returns:
            Dict[str, Any]: 文档信息
        """
        try:
            logger.info(f"获取文档信息: {document_path}")

            # 获取文件基本信息
            file_info = self.document_processor.get_file_info(document_path)

            # 获取存储的分片信息
            chunks_info = self.vector_store.get_document_chunks(document_path)

            result = {
                "document_path": document_path,
                "file_info": file_info,
                "storage_info": {
                    "is_stored": chunks_info["status"] == "success",
                    "total_chunks": chunks_info.get("total_chunks", 0),
                    "chunks": chunks_info.get("chunks", [])
                }
            }

            logger.info(f"文档信息获取完成: {document_path}")
            return result

        except Exception as e:
            error_msg = f"获取文档信息失败: {document_path}, 错误: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessError(error_msg) from e

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

        for document_path in document_paths:
            try:
                result = self.process_document(document_path, chunk_size, chunk_overlap)
                results["processing_results"].append(result)
                results["successful_documents"] += 1

            except Exception as e:
                error_info = {
                    "document_path": document_path,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                results["errors"].append(error_info)
                results["failed_documents"] += 1

                logger.error(f"批量处理中文档失败: {document_path}, 错误: {str(e)}")

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
            documents_info = self.vector_store.list_stored_documents()
            stats["documents"] = documents_info.get("documents", [])
            
        except Exception as e:
            logger.warning(f"获取存储统计信息失败: {str(e)}")

        return stats


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

            # 检查模型加载器
            model_info = self.model_loader.get_model_info()
            health_info["components"]["models"] = {
                "status": "healthy" if model_info["models_loaded"] else "unhealthy",
                "embedding_model_loaded": model_info["embedding_model_loaded"],
                "reranker_model_loaded": model_info["reranker_model_loaded"]
            }

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
            all_healthy = (
                db_health["status"] == "healthy" and
                health_info["components"]["models"]["status"] == "healthy"
            )

            health_info["status"] = "healthy" if all_healthy else "unhealthy"

        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
            logger.error(f"服务健康检查失败: {str(e)}")

        return health_info
