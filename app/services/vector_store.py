"""
向量存储服务模块
使用 LangChain 集成，负责文档的向量化、存储和检索
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from app.core.config import Settings
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import DatabaseError, ModelLoadError

logger = logging.getLogger(__name__)


class VectorStore:
    """
    向量存储服务类

    使用 LangChain 与 ChromaDB 的集成，简化文档存储和检索流程。
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

        try:
            logger.info("初始化 LangChain Chroma 向量存储...")
            embedding_function = self.model_loader.load_embedding_model()

            self.vector_db: Chroma = Chroma(
                client=self.db_service.connect(),
                collection_name=self.settings.COLLECTION_NAME,
                embedding_function=embedding_function
            )
            logger.info("LangChain Chroma 向量存储初始化成功")
        except Exception as e:
            error_msg = f"初始化 LangChain Chroma 实例失败: {e}"
            logger.error(error_msg)
            if isinstance(e, ModelLoadError):
                raise
            raise DatabaseError(error_msg) from e

        logger.info("向量存储服务初始化完成")

    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        将 Document 对象列表添加到向量存储中

        Args:
            documents: LangChain Document 对象列表

        Returns:
            Dict[str, Any]: 存储结果信息

        Raises:
            DatabaseError: 当数据库操作失败时
            ModelLoadError: 当向量化失败时
        """
        if not documents:
            logger.warning("没有文档需要添加，跳过存储")
            return {"chunks_stored": 0, "status": "skipped"}

        try:
            # 过滤掉 LangChain 不支持的复杂元数据类型
            filtered_documents = filter_complex_metadata(documents)

            # 为每个 document 生成一个唯一的 ID
            ids = [str(uuid.uuid4()) for _ in filtered_documents]

            logger.info(f"开始向 ChromaDB 添加 {len(filtered_documents)} 个文档分片...")
            self.vector_db.add_documents(documents=filtered_documents, ids=ids)

            # 从第一个文档的元数据中获取源路径
            doc_path = "Unknown"
            if filtered_documents and filtered_documents[0].metadata and 'source' in filtered_documents[0].metadata:
                doc_path = filtered_documents[0].metadata['source']

            logger.info(f"文档 '{doc_path}' 的 {len(filtered_documents)} 个分片存储完成")

            return {
                "document_path": doc_path,
                "chunks_stored": len(filtered_documents),
                "status": "success",
                "collection_name": self.vector_db._collection.name,
            }
        except Exception as e:
            error_msg = f"通过 LangChain 添加文档失败: {e}"
            logger.error(error_msg)
            if isinstance(e, (DatabaseError, ModelLoadError)):
                raise
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

            # 在 LangChain v0.1+ 中，直接访问 collection
            collection = self.vector_db._collection

            # 查询该文档的所有分片
            results = collection.get(
                where={"source": document_path},
                include=[] # 只需要 ids
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

            collection = self.vector_db._collection

            # 查询该文档的所有分片
            results = collection.get(
                where={"source": document_path},
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

            # 按分片索引排序 (如果元数据中存在)
            # LangChain 的分割器不一定保证 chunk_index, 但我们可以尝试排序
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

            collection = self.vector_db._collection

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
                doc_path = metadata.get("source", "unknown")
                if doc_path not in document_stats:
                    document_stats[doc_path] = {
                        "document_path": doc_path,
                        "chunk_count": 0,
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
