"""
ChromaDB 数据库服务模块
负责 ChromaDB 连接管理、集合创建和基础数据库操作
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import functools

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings
from app.core.exceptions import DatabaseError


logger = logging.getLogger(__name__)


def _ensure_connected(func):
    """
    装饰器：确保在执行方法前 ChromaDB 已连接。
    """
    @functools.wraps(func)
    def wrapper(self: 'ChromaDBService', *args, **kwargs):
        if not self.is_connected():
            self.connect()
        return func(self, *args, **kwargs)
    return wrapper


class ChromaDBService:
    """
    ChromaDB 数据库服务类

    通过装饰器和优化的方法，提供更简洁、健壮的数据库操作。
    """

    def __init__(self, settings: Settings):
        """
        初始化 ChromaDB 服务

        Args:
            settings: 系统配置对象
        """
        self.settings = settings
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None
        self._is_connected = False
        logger.info(f"ChromaDB 服务初始化，数据库路径: {self.settings.chroma_db_full_path}")

    def connect(self) -> chromadb.ClientAPI:
        """
        连接到 ChromaDB 数据库

        Returns:
            chromadb.ClientAPI: ChromaDB 客户端实例

        Raises:
            DatabaseError: 当数据库连接失败时抛出异常
        """
        if self.is_connected():
            logger.debug("ChromaDB 已连接，返回现有客户端")
            return self.client

        try:
            db_path = self.settings.chroma_db_full_path
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"正在连接到 ChromaDB: {db_path}")

            chroma_settings = ChromaSettings(
                persist_directory=str(db_path),
                is_persistent=True,
                anonymized_telemetry=False
            )
            self.client = chromadb.PersistentClient(path=str(db_path), settings=chroma_settings)
            self._is_connected = True
            logger.info(f"ChromaDB 连接成功: {db_path}")
            return self.client
        except Exception as e:
            error_msg = f"ChromaDB 连接失败: {str(e)}"
            logger.error(error_msg)
            self.client = None
            self._is_connected = False
            raise DatabaseError(error_msg) from e

    def disconnect(self):
        """
        断开 ChromaDB 连接。
        """
        self.client = None
        self.collection = None
        self._is_connected = False
        logger.info("ChromaDB 连接已断开")

    def is_connected(self) -> bool:
        """
        检查数据库连接状态

        Returns:
            bool: 连接状态
        """
        return self._is_connected and self.client is not None

    @_ensure_connected
    def create_or_get_collection(self, collection_name: Optional[str] = None) -> Collection:
        """
        创建或获取文档集合。此方法现在更健壮，并处理了默认集合的缓存。

        Args:
            collection_name: 集合名称，如果为 None 则使用配置中的默认名称

        Returns:
            Collection: ChromaDB 集合对象

        Raises:
            DatabaseError: 当集合操作失败时抛出异常
        """
        name = collection_name or self.settings.COLLECTION_NAME

        # 如果请求的是已缓存的默认集合，直接返回
        if name == self.settings.COLLECTION_NAME and self.collection:
            return self.collection

        try:
            # 尝试获取现有集合
            collection = self.client.get_collection(name=name)
            logger.info(f"获取到现有集合: {name}")
        except ValueError:
            # 集合不存在时，ChromaDB 抛出 ValueError
            logger.info(f"集合 '{name}' 不存在，创建新集合")
            try:
                collection = self.client.create_collection(
                    name=name,
                    metadata={"description": "Local RAG 文档向量存储集合"}
                )
                logger.info(f"集合 '{name}' 创建成功")
            except Exception as e:
                error_msg = f"创建集合 '{name}' 失败: {e}"
                logger.error(error_msg)
                raise DatabaseError(error_msg) from e

        # 如果操作的是默认集合，则更新实例上的缓存
        if name == self.settings.COLLECTION_NAME:
            self.collection = collection
        return collection

    @_ensure_connected
    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取集合信息

        Args:
            collection_name: 集合名称，如果为 None 则使用默认集合

        Returns:
            Dict[str, Any]: 包含集合信息的字典

        Raises:
            DatabaseError: 当获取集合信息失败时抛出异常
        """
        name = collection_name or self.settings.COLLECTION_NAME
        try:
            # 使用 create_or_get_collection 确保集合存在
            collection = self.create_or_get_collection(name)
            info = {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata,
                "id": collection.id
            }
            logger.info(f"集合信息: {info}")
            return info
        except Exception as e:
            error_msg = f"获取集合 '{name}' 信息失败: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    @_ensure_connected
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        列出所有集合

        Returns:
            List[Dict[str, Any]]: 包含所有集合信息的列表

        Raises:
            DatabaseError: 当列出集合失败时抛出异常
        """
        try:
            collections = self.client.list_collections()
            collection_info = [
                {
                    "name": c.name,
                    "id": c.id,
                    "metadata": c.metadata,
                    "count": c.count()
                }
                for c in collections
            ]
            logger.info(f"找到 {len(collection_info)} 个集合")
            return collection_info
        except Exception as e:
            error_msg = f"列出集合失败: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    @_ensure_connected
    def delete_collection(self, collection_name: str) -> bool:
        """
        删除指定集合

        Args:
            collection_name: 要删除的集合名称

        Returns:
            bool: 删除成功返回 True

        Raises:
            DatabaseError: 当删除集合失败时抛出异常
        """
        try:
            self.client.delete_collection(name=collection_name)
            # 如果删除的是当前缓存的集合，清理引用
            if self.collection and self.collection.name == collection_name:
                self.collection = None
            logger.info(f"集合 '{collection_name}' 删除成功")
            return True
        except Exception as e:
            error_msg = f"删除集合 '{collection_name}' 失败: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    @_ensure_connected
    def reset_collection(self, collection_name: Optional[str] = None) -> Collection:
        """
        重置集合（先删除后创建）

        Args:
            collection_name: 集合名称，如果为 None 则使用默认集合

        Returns:
            Collection: 重置后的集合对象

        Raises:
            DatabaseError: 当重置集合失败时抛出异常
        """
        name = collection_name or self.settings.COLLECTION_NAME
        logger.info(f"开始重置集合: {name}")
        try:
            # 尝试删除，即使集合不存在也继续
            self.delete_collection(name)
        except Exception as e:
            # 记录删除失败的警告，但这不应阻止创建
            logger.warning(f"重置集合时删除步骤失败 (可能集合不存在，可忽略): {e}")

        collection = self.create_or_get_collection(name)
        logger.info(f"集合 '{name}' 重置完成")
        return collection

    @_ensure_connected
    def list_documents(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出集合中所有唯一的文档及其信息

        Args:
            collection_name: 集合名称，如果为 None 则使用默认集合

        Returns:
            List[Dict[str, Any]]: 包含每个文档信息的列表

        Raises:
            DatabaseError: 当操作失败时
        """
        name = collection_name or self.settings.COLLECTION_NAME
        try:
            collection = self.create_or_get_collection(name)
            if collection.count() == 0:
                return []

            results = collection.get(include=['metadatas'])
            if not results or not results['metadatas']:
                return []

            documents_summary = {}
            for metadata in results['metadatas']:
                doc_path = metadata.get('source')
                if not doc_path:
                    continue

                if doc_path not in documents_summary:
                    documents_summary[doc_path] = {
                        "document_path": doc_path,
                        "chunk_count": 0,
                        "created_at": metadata.get('created_at'),
                        "text_length": metadata.get('text_length', 0),
                        "chunk_size": metadata.get('chunk_size', 0),
                        "file_size": metadata.get('file_size', 0)
                    }
                documents_summary[doc_path]['chunk_count'] += 1
                # 持续更新，以防元数据不一致，确保取到有效值
                if documents_summary[doc_path]['text_length'] == 0:
                    documents_summary[doc_path]['text_length'] = metadata.get('text_length', 0)
                if documents_summary[doc_path]['chunk_size'] == 0:
                    documents_summary[doc_path]['chunk_size'] = metadata.get('chunk_size', 0)
                if documents_summary[doc_path]['file_size'] == 0:
                    documents_summary[doc_path]['file_size'] = metadata.get('file_size', 0)

            logger.info(f"在集合 '{name}' 中找到 {len(documents_summary)} 个独立文档")
            return list(documents_summary.values())
        except Exception as e:
            error_msg = f"列出集合 '{name}' 中的文档失败: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库信息

        Returns:
            Dict[str, Any]: 包含数据库信息的字典
        """
        info = {
            "database_path": str(self.settings.chroma_db_full_path),
            "is_connected": self.is_connected(),
            "default_collection": self.settings.COLLECTION_NAME,
            "current_collection": self.collection.name if self.collection else None
        }
        if self.is_connected():
            try:
                collections = self.list_collections()
                info["collections"] = collections
                info["total_collections"] = len(collections)
            except Exception as e:
                logger.warning(f"获取集合列表时发生警告: {e}")
                info["collections"] = []
                info["total_collections"] = 0
        return info

    def health_check(self) -> Dict[str, Any]:
        """
        数据库健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        health_info = {
            "status": "unhealthy",
            "database_path_exists": False,
            "connection_status": False,
            "default_collection_exists": False,
            "error": None
        }
        try:
            db_path = self.settings.chroma_db_full_path
            health_info["database_path_exists"] = db_path.exists()

            # 尝试连接，会通过 is_connected() 更新状态
            self.connect()
            health_info["connection_status"] = self.is_connected()

            if self.is_connected():
                # 尝试获取默认集合
                self.create_or_get_collection()
                health_info["default_collection_exists"] = True

            if all([health_info["database_path_exists"],
                    health_info["connection_status"],
                    health_info["default_collection_exists"]]):
                health_info["status"] = "healthy"
        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
            logger.error(f"数据库健康检查失败: {e}")

        return health_info
