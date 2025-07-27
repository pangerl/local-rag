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
from app.core.chunker import JiebaChunker
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
        self.chunker = JiebaChunker()
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
            logger.info(f"开始处理文档: {document_path}")
            
            # 1. 验证文件存在性和格式
            self._validate_document(document_path)
            
            # 2. 提取文档文本内容
            text_content = self._extract_document_text(document_path)
            
            # 3. 文本分片处理
            chunks = self._chunk_document_text(
                text_content, 
                chunk_size or self.settings.DEFAULT_CHUNK_SIZE,
                chunk_overlap or self.settings.DEFAULT_CHUNK_OVERLAP
            )
            
            # 4. 向量化和存储
            storage_result = self._store_document_chunks(document_path, chunks)
            
            # 5. 更新统计信息
            processing_time = time.time() - start_time
            self._update_processing_stats(len(chunks), processing_time)
            
            result = {
                "document_path": document_path,
                "status": "success",
                "text_length": len(text_content),
                "chunks_created": len(chunks),
                "chunks_stored": storage_result.get("chunks_stored", 0),
                "processing_time": processing_time,
                "chunk_size": chunk_size or self.settings.DEFAULT_CHUNK_SIZE,
                "chunk_overlap": chunk_overlap or self.settings.DEFAULT_CHUNK_OVERLAP,
                "embedding_dimension": storage_result.get("embedding_dimension"),
                "collection_name": storage_result.get("collection_name")
            }
            
            logger.info(f"文档处理完成: {document_path}, "
                       f"耗时: {processing_time:.2f}s, "
                       f"分片数: {len(chunks)}")
            
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
            logger.info(f"开始更新文档: {document_path}")
            
            # 1. 验证文件存在性和格式
            self._validate_document(document_path)
            
            # 2. 提取文档文本内容
            text_content = self._extract_document_text(document_path)
            
            # 3. 文本分片处理
            chunks = self._chunk_document_text(
                text_content, 
                chunk_size or self.settings.DEFAULT_CHUNK_SIZE,
                chunk_overlap or self.settings.DEFAULT_CHUNK_OVERLAP
            )
            
            # 4. 更新向量存储
            update_result = self.vector_store.update_document_chunks(document_path, chunks)
            
            # 5. 更新统计信息
            processing_time = time.time() - start_time
            self._update_processing_stats(len(chunks), processing_time)
            
            result = {
                "document_path": document_path,
                "status": "success",
                "text_length": len(text_content),
                "old_chunks_deleted": update_result.get("old_chunks_deleted", 0),
                "new_chunks_stored": update_result.get("new_chunks_stored", 0),
                "processing_time": processing_time,
                "chunk_size": chunk_size or self.settings.DEFAULT_CHUNK_SIZE,
                "chunk_overlap": chunk_overlap or self.settings.DEFAULT_CHUNK_OVERLAP
            }
            
            logger.info(f"文档更新完成: {document_path}, "
                       f"耗时: {processing_time:.2f}s, "
                       f"新分片数: {len(chunks)}")
            
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
        except Exception as e:
            logger.warning(f"获取存储统计信息失败: {str(e)}")
        
        return stats
    
    def _validate_document(self, document_path: str):
        """
        验证文档存在性和格式
        
        Args:
            document_path: 文档路径
            
        Raises:
            FileNotFoundError: 当文件不存在时抛出异常
            UnsupportedFormatError: 当文件格式不支持时抛出异常
        """
        if not Path(document_path).exists():
            error_msg = f"文件不存在: {document_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not self.document_processor.validate_file_format(document_path):
            error_msg = f"不支持的文件格式: {document_path}"
            logger.error(error_msg)
            raise UnsupportedFormatError(error_msg)
    
    def _extract_document_text(self, document_path: str) -> str:
        """
        提取文档文本内容
        
        Args:
            document_path: 文档路径
            
        Returns:
            str: 提取的文本内容
            
        Raises:
            DocumentProcessError: 当文本提取失败时抛出异常
        """
        success, text_content = self.document_processor.extract_text(document_path)
        
        if not success:
            error_msg = f"文本提取失败: {document_path}"
            logger.error(error_msg)
            raise DocumentProcessError(error_msg)
        
        if not text_content.strip():
            error_msg = f"文档内容为空: {document_path}"
            logger.error(error_msg)
            raise DocumentProcessError(error_msg)
        
        logger.info(f"文本提取成功: {document_path}, 长度: {len(text_content)}")
        return text_content
    
    def _chunk_document_text(self, text_content: str, chunk_size: int, 
                           chunk_overlap: int) -> List[str]:
        """
        对文档文本进行分片处理
        
        Args:
            text_content: 文档文本内容
            chunk_size: 分片大小（词元数量）
            chunk_overlap: 分片重叠（词元数量）
            
        Returns:
            List[str]: 文档分片列表
            
        Raises:
            DocumentProcessError: 当分片处理失败时抛出异常
        """
        try:
            chunks = self.chunker.chunk_text(text_content, chunk_size, chunk_overlap)
            
            if not chunks:
                error_msg = f"文本分片结果为空，原文长度: {len(text_content)}"
                logger.error(error_msg)
                raise DocumentProcessError(error_msg)
            
            logger.info(f"文本分片完成，生成 {len(chunks)} 个分片")
            return chunks
            
        except Exception as e:
            error_msg = f"文本分片失败: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessError(error_msg) from e
    
    def _store_document_chunks(self, document_path: str, chunks: List[str]) -> Dict[str, Any]:
        """
        存储文档分片到向量数据库
        
        Args:
            document_path: 文档路径
            chunks: 文档分片列表
            
        Returns:
            Dict[str, Any]: 存储结果
            
        Raises:
            DatabaseError: 当存储失败时抛出异常
            ModelLoadError: 当向量化失败时抛出异常
        """
        try:
            storage_result = self.vector_store.store_document_chunks(document_path, chunks)
            
            if storage_result["status"] != "success":
                error_msg = f"文档分片存储失败: {document_path}, 结果: {storage_result}"
                logger.error(error_msg)
                raise DatabaseError(error_msg)
            
            logger.info(f"文档分片存储成功: {document_path}, "
                       f"存储分片数: {storage_result['chunks_stored']}")
            
            return storage_result
            
        except Exception as e:
            if isinstance(e, (DatabaseError, ModelLoadError)):
                raise
            else:
                error_msg = f"文档分片存储过程失败: {document_path}, 错误: {str(e)}"
                logger.error(error_msg)
                raise DatabaseError(error_msg) from e
    
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
            health_info["components"]["chunker"] = {
                "status": "healthy",
                "jieba_initialized": hasattr(self.chunker, 'jieba')
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