"""
ChromaDB 数据库服务模块
负责 ChromaDB 连接管理、集合创建和基础数据库操作
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings
from app.core.exceptions import DatabaseError


logger = logging.getLogger(__name__)


class ChromaDBService:
    """
    ChromaDB 数据库服务类
    
    负责管理 ChromaDB 客户端连接、集合创建和基础数据库操作
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
        if self.client is not None and self._is_connected:
            logger.info("ChromaDB 已连接，返回现有客户端")
            return self.client
        
        try:
            # 确保数据库目录存在
            db_path = self.settings.chroma_db_full_path
            db_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"正在连接到 ChromaDB: {db_path}")
            
            # 创建 ChromaDB 客户端配置
            chroma_settings = ChromaSettings(
                persist_directory=str(db_path),
                is_persistent=True,
                anonymized_telemetry=False  # 禁用遥测数据收集
            )
            
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=chroma_settings
            )
            
            self._is_connected = True
            logger.info(f"ChromaDB 连接成功: {db_path}")
            
            return self.client
            
        except Exception as e:
            error_msg = f"ChromaDB 连接失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def disconnect(self):
        """
        断开 ChromaDB 连接
        """
        if self.client is not None:
            try:
                # ChromaDB 客户端没有显式的关闭方法，只需要清理引用
                self.client = None
                self.collection = None
                self._is_connected = False
                logger.info("ChromaDB 连接已断开")
            except Exception as e:
                logger.warning(f"断开 ChromaDB 连接时发生警告: {str(e)}")
    
    def create_or_get_collection(self, collection_name: Optional[str] = None) -> Collection:
        """
        创建或获取文档集合
        
        Args:
            collection_name: 集合名称，如果为 None 则使用配置中的默认名称
            
        Returns:
            Collection: ChromaDB 集合对象
            
        Raises:
            DatabaseError: 当集合操作失败时抛出异常
        """
        if not self._is_connected or self.client is None:
            self.connect()
        
        collection_name = collection_name or self.settings.COLLECTION_NAME
        
        try:
            logger.info(f"创建或获取集合: {collection_name}")
            
            # 尝试获取现有集合
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.info(f"获取到现有集合: {collection_name}")
            except Exception:
                # 集合不存在，创建新集合
                logger.info(f"集合不存在，创建新集合: {collection_name}")
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Local RAG 文档向量存储集合"}
                )
                logger.info(f"集合创建成功: {collection_name}")
            
            self.collection = collection
            return collection
            
        except Exception as e:
            error_msg = f"集合操作失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取集合信息
        
        Args:
            collection_name: 集合名称，如果为 None 则使用当前集合
            
        Returns:
            Dict[str, Any]: 包含集合信息的字典
            
        Raises:
            DatabaseError: 当获取集合信息失败时抛出异常
        """
        try:
            if collection_name:
                if not self._is_connected or self.client is None:
                    self.connect()
                collection = self.client.get_collection(name=collection_name)
            elif self.collection:
                collection = self.collection
            else:
                collection = self.create_or_get_collection()
            
            # 获取集合统计信息
            count = collection.count()
            
            info = {
                "name": collection.name,
                "count": count,
                "metadata": collection.metadata,
                "id": collection.id
            }
            
            logger.info(f"集合信息: {info}")
            return info
            
        except Exception as e:
            error_msg = f"获取集合信息失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        列出所有集合
        
        Returns:
            List[Dict[str, Any]]: 包含所有集合信息的列表
            
        Raises:
            DatabaseError: 当列出集合失败时抛出异常
        """
        if not self._is_connected or self.client is None:
            self.connect()
        
        try:
            collections = self.client.list_collections()
            
            collection_info = []
            for collection in collections:
                info = {
                    "name": collection.name,
                    "id": collection.id,
                    "metadata": collection.metadata,
                    "count": collection.count()
                }
                collection_info.append(info)
            
            logger.info(f"找到 {len(collection_info)} 个集合")
            return collection_info
            
        except Exception as e:
            error_msg = f"列出集合失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
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
        if not self._is_connected or self.client is None:
            self.connect()
        
        try:
            logger.info(f"删除集合: {collection_name}")
            
            self.client.delete_collection(name=collection_name)
            
            # 如果删除的是当前集合，清理引用
            if self.collection and self.collection.name == collection_name:
                self.collection = None
            
            logger.info(f"集合删除成功: {collection_name}")
            return True
            
        except Exception as e:
            error_msg = f"删除集合失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def reset_collection(self, collection_name: Optional[str] = None) -> Collection:
        """
        重置集合（删除所有数据）
        
        Args:
            collection_name: 集合名称，如果为 None 则使用当前集合
            
        Returns:
            Collection: 重置后的集合对象
            
        Raises:
            DatabaseError: 当重置集合失败时抛出异常
        """
        collection_name = collection_name or self.settings.COLLECTION_NAME
        
        try:
            logger.info(f"重置集合: {collection_name}")
            
            # 删除现有集合
            try:
                self.delete_collection(collection_name)
            except DatabaseError:
                # 集合可能不存在，忽略错误
                pass
            
            # 创建新集合
            collection = self.create_or_get_collection(collection_name)
            
            logger.info(f"集合重置完成: {collection_name}")
            return collection
            
        except Exception as e:
            error_msg = f"重置集合失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def is_connected(self) -> bool:
        """
        检查数据库连接状态
        
        Returns:
            bool: 连接状态
        """
        return self._is_connected and self.client is not None
    
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
                logger.warning(f"获取集合列表时发生警告: {str(e)}")
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
            "status": "unknown",
            "database_path_exists": False,
            "connection_status": False,
            "default_collection_exists": False,
            "error": None
        }
        
        try:
            # 检查数据库路径
            db_path = self.settings.chroma_db_full_path
            health_info["database_path_exists"] = db_path.exists()
            
            # 检查连接状态
            if not self.is_connected():
                self.connect()
            health_info["connection_status"] = self.is_connected()
            
            # 检查默认集合
            if self.is_connected():
                try:
                    self.create_or_get_collection()
                    health_info["default_collection_exists"] = True
                except Exception as e:
                    health_info["error"] = f"集合检查失败: {str(e)}"
            
            # 确定整体状态
            if (health_info["database_path_exists"] and 
                health_info["connection_status"] and 
                health_info["default_collection_exists"]):
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "unhealthy"
                
        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
            logger.error(f"数据库健康检查失败: {str(e)}")
        
        return health_info