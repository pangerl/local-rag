"""
向量存储服务模块
负责文档分片的向量化、存储和元数据管理
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import DatabaseError, ModelLoadError


logger = logging.getLogger(__name__)


class VectorStore:
    """
    向量存储服务类
    
    负责文档分片的向量化、存储和元数据管理
    """
    
    def __init__(self, settings: Settings, db_service: ChromaDBService, model_loader: ModelLoader):
        """
        初始化向量存储服务
        
        Args:
            settings: 系统配置对象
            db_service: ChromaDB 数据库服务
            model_loader: 模型加载器
        """
        self.settings = settings
        self.db_service = db_service
        self.model_loader = model_loader
        self.embedding_model: Optional[SentenceTransformer] = None
        
        logger.info("向量存储服务初始化完成")
    
    def _ensure_embedding_model(self) -> SentenceTransformer:
        """
        确保嵌入模型已加载
        
        Returns:
            SentenceTransformer: 嵌入模型实例
            
        Raises:
            ModelLoadError: 当模型加载失败时抛出异常
        """
        if self.embedding_model is None:
            try:
                logger.info("加载嵌入模型用于向量化")
                self.embedding_model = self.model_loader.load_embedding_model()
            except Exception as e:
                error_msg = f"嵌入模型加载失败: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg) from e
        
        return self.embedding_model
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 嵌入向量数组
            
        Raises:
            ModelLoadError: 当向量化失败时抛出异常
        """
        try:
            model = self._ensure_embedding_model()
            logger.info(f"开始向量化 {len(texts)} 个文本片段")
            
            # 生成嵌入向量
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # 归一化向量，提高检索效果
            )
            
            logger.info(f"向量化完成，生成 {embeddings.shape} 形状的嵌入矩阵")
            return embeddings
            
        except Exception as e:
            error_msg = f"文本向量化失败: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def _prepare_metadata(self, document_path: str, chunk_index: int, 
                         chunk_text: str, total_chunks: int) -> Dict[str, Any]:
        """
        准备文档分片元数据
        
        Args:
            document_path: 文档路径
            chunk_index: 分片索引
            chunk_text: 分片文本
            total_chunks: 总分片数
            
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = {
            "document_path": document_path,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_length": len(chunk_text),
            "created_at": datetime.now().isoformat(),
            "chunk_id": f"{document_path}_{chunk_index}"
        }
        
        return metadata
    
    def store_document_chunks(self, document_path: str, chunks: List[str]) -> Dict[str, Any]:
        """
        存储文档分片到向量数据库
        
        Args:
            document_path: 文档路径
            chunks: 文档分片列表
            
        Returns:
            Dict[str, Any]: 存储结果信息
            
        Raises:
            DatabaseError: 当数据库操作失败时抛出异常
            ModelLoadError: 当向量化失败时抛出异常
        """
        if not chunks:
            logger.warning(f"文档 {document_path} 没有分片，跳过存储")
            return {
                "document_path": document_path,
                "chunks_stored": 0,
                "status": "skipped",
                "message": "没有分片需要存储"
            }
        
        try:
            logger.info(f"开始存储文档 {document_path} 的 {len(chunks)} 个分片")
            
            # 生成嵌入向量
            embeddings = self._generate_embeddings(chunks)
            
            # 获取或创建集合
            collection = self.db_service.create_or_get_collection()
            
            # 准备存储数据
            ids = []
            metadatas = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                # 生成唯一 ID
                chunk_id = f"{document_path}_{i}_{uuid.uuid4().hex[:8]}"
                ids.append(chunk_id)
                
                # 准备元数据
                metadata = self._prepare_metadata(document_path, i, chunk, len(chunks))
                metadatas.append(metadata)
                
                # 存储原始文本
                documents.append(chunk)
            
            # 批量存储到 ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"文档 {document_path} 的 {len(chunks)} 个分片存储完成")
            
            return {
                "document_path": document_path,
                "chunks_stored": len(chunks),
                "status": "success",
                "collection_name": collection.name,
                "embedding_dimension": embeddings.shape[1]
            }
            
        except Exception as e:
            error_msg = f"存储文档分片失败: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (DatabaseError, ModelLoadError)):
                raise
            else:
                raise DatabaseError(error_msg) from e
    
    def update_document_chunks(self, document_path: str, chunks: List[str]) -> Dict[str, Any]:
        """
        更新文档分片（先删除旧分片，再存储新分片）
        
        Args:
            document_path: 文档路径
            chunks: 新的文档分片列表
            
        Returns:
            Dict[str, Any]: 更新结果信息
        """
        try:
            logger.info(f"开始更新文档 {document_path} 的分片")
            
            # 删除旧分片
            delete_result = self.delete_document_chunks(document_path)
            
            # 存储新分片
            store_result = self.store_document_chunks(document_path, chunks)
            
            logger.info(f"文档 {document_path} 更新完成")
            
            return {
                "document_path": document_path,
                "old_chunks_deleted": delete_result.get("chunks_deleted", 0),
                "new_chunks_stored": store_result.get("chunks_stored", 0),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"更新文档分片失败: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (DatabaseError, ModelLoadError)):
                raise
            else:
                raise DatabaseError(error_msg) from e
    
    def delete_document_chunks(self, document_path: str) -> Dict[str, Any]:
        """
        删除指定文档的所有分片
        
        Args:
            document_path: 文档路径
            
        Returns:
            Dict[str, Any]: 删除结果信息
        """
        try:
            logger.info(f"开始删除文档 {document_path} 的分片")
            
            # 获取集合
            collection = self.db_service.create_or_get_collection()
            
            # 查询该文档的所有分片
            results = collection.get(
                where={"document_path": document_path},
                include=["metadatas"]
            )
            
            if not results["ids"]:
                logger.info(f"文档 {document_path} 没有找到分片，无需删除")
                return {
                    "document_path": document_path,
                    "chunks_deleted": 0,
                    "status": "not_found"
                }
            
            # 删除分片
            collection.delete(ids=results["ids"])
            
            chunks_deleted = len(results["ids"])
            logger.info(f"文档 {document_path} 的 {chunks_deleted} 个分片删除完成")
            
            return {
                "document_path": document_path,
                "chunks_deleted": chunks_deleted,
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"删除文档分片失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def get_document_chunks(self, document_path: str) -> Dict[str, Any]:
        """
        获取指定文档的所有分片信息
        
        Args:
            document_path: 文档路径
            
        Returns:
            Dict[str, Any]: 文档分片信息
        """
        try:
            logger.info(f"获取文档 {document_path} 的分片信息")
            
            # 获取集合
            collection = self.db_service.create_or_get_collection()
            
            # 查询该文档的所有分片
            results = collection.get(
                where={"document_path": document_path},
                include=["metadatas", "documents"]
            )
            
            if not results["ids"]:
                return {
                    "document_path": document_path,
                    "chunks": [],
                    "total_chunks": 0,
                    "status": "not_found"
                }
            
            # 整理分片信息
            chunks = []
            for i, chunk_id in enumerate(results["ids"]):
                chunk_info = {
                    "id": chunk_id,
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                chunks.append(chunk_info)
            
            # 按分片索引排序
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            
            logger.info(f"文档 {document_path} 找到 {len(chunks)} 个分片")
            
            return {
                "document_path": document_path,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"获取文档分片失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def list_stored_documents(self) -> Dict[str, Any]:
        """
        列出所有已存储的文档
        
        Returns:
            Dict[str, Any]: 存储的文档列表信息
        """
        try:
            logger.info("获取所有已存储文档列表")
            
            # 获取集合
            collection = self.db_service.create_or_get_collection()
            
            # 获取所有元数据
            results = collection.get(include=["metadatas"])
            
            if not results["metadatas"]:
                return {
                    "documents": [],
                    "total_documents": 0,
                    "total_chunks": 0,
                    "status": "empty"
                }
            
            # 统计文档信息
            document_stats = {}
            for metadata in results["metadatas"]:
                doc_path = metadata.get("document_path", "unknown")
                if doc_path not in document_stats:
                    document_stats[doc_path] = {
                        "document_path": doc_path,
                        "chunk_count": 0,
                        "total_chunks": metadata.get("total_chunks", 0),
                        "created_at": metadata.get("created_at", "unknown")
                    }
                document_stats[doc_path]["chunk_count"] += 1
            
            documents = list(document_stats.values())
            documents.sort(key=lambda x: x["created_at"], reverse=True)
            
            logger.info(f"找到 {len(documents)} 个已存储文档，共 {len(results['metadatas'])} 个分片")
            
            return {
                "documents": documents,
                "total_documents": len(documents),
                "total_chunks": len(results["metadatas"]),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"获取文档列表失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            Dict[str, Any]: 存储统计信息
        """
        try:
            logger.info("获取存储统计信息")
            
            # 获取集合信息
            collection_info = self.db_service.get_collection_info()
            
            # 获取文档列表
            documents_info = self.list_stored_documents()
            
            # 计算统计信息
            stats = {
                "collection_name": collection_info["name"],
                "total_chunks": collection_info["count"],
                "total_documents": documents_info["total_documents"],
                "average_chunks_per_document": (
                    collection_info["count"] / documents_info["total_documents"] 
                    if documents_info["total_documents"] > 0 else 0
                ),
                "database_path": str(self.db_service.settings.chroma_db_full_path),
                "embedding_model": str(self.settings.embedding_model_path),
                "status": "success"
            }
            
            logger.info(f"存储统计: {stats['total_documents']} 个文档，{stats['total_chunks']} 个分片")
            
            return stats
            
        except Exception as e:
            error_msg = f"获取存储统计失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def clear_all_data(self) -> Dict[str, Any]:
        """
        清空所有存储数据
        
        Returns:
            Dict[str, Any]: 清空结果信息
        """
        try:
            logger.warning("开始清空所有存储数据")
            
            # 重置集合（删除所有数据）
            collection = self.db_service.reset_collection()
            
            logger.warning("所有存储数据已清空")
            
            return {
                "status": "success",
                "message": "所有数据已清空",
                "collection_name": collection.name
            }
            
        except Exception as e:
            error_msg = f"清空数据失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e