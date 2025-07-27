"""
ChromaDB 数据库服务单元测试
测试 ChromaDBService 的连接管理和集合操作功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import Settings
from app.services.database import ChromaDBService
from app.core.exceptions import DatabaseError


class TestChromaDBService:
    """ChromaDBService 测试类"""
    
    @pytest.fixture
    def temp_db_dir(self):
        """创建临时数据库目录用于测试"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_settings(self, temp_db_dir):
        """创建测试用的配置对象"""
        settings = Settings()
        settings.CHROMA_DB_PATH = temp_db_dir
        settings.COLLECTION_NAME = "test_documents"
        return settings
    
    @pytest.fixture
    def db_service(self, test_settings):
        """创建 ChromaDBService 实例"""
        return ChromaDBService(test_settings)
    
    def test_init(self, db_service, test_settings):
        """测试 ChromaDBService 初始化"""
        assert db_service.settings == test_settings
        assert db_service.client is None
        assert db_service.collection is None
        assert db_service._is_connected is False
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_connect_success(self, mock_persistent_client, db_service):
        """测试数据库连接成功"""
        # 创建模拟客户端
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_persistent_client.return_value = mock_client
        
        result = db_service.connect()
        
        assert result == mock_client
        assert db_service.client == mock_client
        assert db_service._is_connected is True
        
        # 验证客户端创建参数
        mock_persistent_client.assert_called_once()
        call_args = mock_persistent_client.call_args
        assert str(db_service.settings.chroma_db_full_path) in call_args[1]['path']
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_connect_already_connected(self, mock_persistent_client, db_service):
        """测试已连接状态下的连接调用"""
        # 设置已连接状态
        mock_client = Mock(spec=chromadb.ClientAPI)
        db_service.client = mock_client
        db_service._is_connected = True
        
        result = db_service.connect()
        
        assert result == mock_client
        # 验证没有创建新客户端
        mock_persistent_client.assert_not_called()
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_connect_failure(self, mock_persistent_client, db_service):
        """测试数据库连接失败"""
        # 模拟连接失败
        mock_persistent_client.side_effect = Exception("连接失败")
        
        with pytest.raises(DatabaseError) as exc_info:
            db_service.connect()
        
        assert "ChromaDB 连接失败" in str(exc_info.value)
        assert db_service._is_connected is False
    
    def test_disconnect(self, db_service):
        """测试数据库断开连接"""
        # 设置连接状态
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_collection = Mock(spec=Collection)
        db_service.client = mock_client
        db_service.collection = mock_collection
        db_service._is_connected = True
        
        db_service.disconnect()
        
        assert db_service.client is None
        assert db_service.collection is None
        assert db_service._is_connected is False
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_create_or_get_collection_new(self, mock_persistent_client, db_service):
        """测试创建新集合"""
        # 设置模拟客户端
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_collection = Mock(spec=Collection)
        mock_persistent_client.return_value = mock_client
        
        # 模拟集合不存在，需要创建
        mock_client.get_collection.side_effect = Exception("集合不存在")
        mock_client.create_collection.return_value = mock_collection
        
        result = db_service.create_or_get_collection()
        
        assert result == mock_collection
        assert db_service.collection == mock_collection
        
        # 验证调用
        mock_client.get_collection.assert_called_once_with(name=db_service.settings.COLLECTION_NAME)
        mock_client.create_collection.assert_called_once()
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_create_or_get_collection_existing(self, mock_persistent_client, db_service):
        """测试获取现有集合"""
        # 设置模拟客户端
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_collection = Mock(spec=Collection)
        mock_persistent_client.return_value = mock_client
        
        # 模拟集合已存在
        mock_client.get_collection.return_value = mock_collection
        
        result = db_service.create_or_get_collection()
        
        assert result == mock_collection
        assert db_service.collection == mock_collection
        
        # 验证只调用了 get_collection
        mock_client.get_collection.assert_called_once_with(name=db_service.settings.COLLECTION_NAME)
        mock_client.create_collection.assert_not_called()
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_create_or_get_collection_custom_name(self, mock_persistent_client, db_service):
        """测试使用自定义集合名称"""
        # 设置模拟客户端
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_collection = Mock(spec=Collection)
        mock_persistent_client.return_value = mock_client
        
        mock_client.get_collection.return_value = mock_collection
        
        custom_name = "custom_collection"
        result = db_service.create_or_get_collection(custom_name)
        
        assert result == mock_collection
        mock_client.get_collection.assert_called_once_with(name=custom_name)
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_create_or_get_collection_failure(self, mock_persistent_client, db_service):
        """测试集合操作失败"""
        # 设置模拟客户端
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_persistent_client.return_value = mock_client
        
        # 模拟获取和创建都失败
        mock_client.get_collection.side_effect = Exception("获取失败")
        mock_client.create_collection.side_effect = Exception("创建失败")
        
        with pytest.raises(DatabaseError) as exc_info:
            db_service.create_or_get_collection()
        
        assert "集合操作失败" in str(exc_info.value)
    
    def test_get_collection_info_with_current_collection(self, db_service):
        """测试获取当前集合信息"""
        # 设置模拟集合
        mock_collection = Mock(spec=Collection)
        mock_collection.name = "test_collection"
        mock_collection.id = "test_id"
        mock_collection.metadata = {"test": "metadata"}
        mock_collection.count.return_value = 100
        
        db_service.collection = mock_collection
        
        result = db_service.get_collection_info()
        
        assert result["name"] == "test_collection"
        assert result["id"] == "test_id"
        assert result["metadata"] == {"test": "metadata"}
        assert result["count"] == 100
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_get_collection_info_with_name(self, mock_persistent_client, db_service):
        """测试通过名称获取集合信息"""
        # 设置模拟客户端和集合
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_collection = Mock(spec=Collection)
        mock_collection.name = "named_collection"
        mock_collection.id = "named_id"
        mock_collection.metadata = {}
        mock_collection.count.return_value = 50
        
        mock_persistent_client.return_value = mock_client
        mock_client.get_collection.return_value = mock_collection
        
        # 设置连接状态
        db_service.client = mock_client
        db_service._is_connected = True
        
        result = db_service.get_collection_info("named_collection")
        
        assert result["name"] == "named_collection"
        assert result["count"] == 50
        mock_client.get_collection.assert_called_once_with(name="named_collection")
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_list_collections(self, mock_persistent_client, db_service):
        """测试列出所有集合"""
        # 设置模拟客户端和集合
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_persistent_client.return_value = mock_client
        
        mock_collection1 = Mock(spec=Collection)
        mock_collection1.name = "collection1"
        mock_collection1.id = "id1"
        mock_collection1.metadata = {"type": "test"}
        mock_collection1.count.return_value = 10
        
        mock_collection2 = Mock(spec=Collection)
        mock_collection2.name = "collection2"
        mock_collection2.id = "id2"
        mock_collection2.metadata = {}
        mock_collection2.count.return_value = 20
        
        mock_client.list_collections.return_value = [mock_collection1, mock_collection2]
        
        result = db_service.list_collections()
        
        assert len(result) == 2
        assert result[0]["name"] == "collection1"
        assert result[0]["count"] == 10
        assert result[1]["name"] == "collection2"
        assert result[1]["count"] == 20
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_delete_collection(self, mock_persistent_client, db_service):
        """测试删除集合"""
        # 设置模拟客户端
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_persistent_client.return_value = mock_client
        
        result = db_service.delete_collection("test_collection")
        
        assert result is True
        mock_client.delete_collection.assert_called_once_with(name="test_collection")
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_delete_collection_current(self, mock_persistent_client, db_service):
        """测试删除当前集合"""
        # 设置模拟客户端和当前集合
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_collection = Mock(spec=Collection)
        mock_collection.name = "current_collection"
        
        mock_persistent_client.return_value = mock_client
        db_service.collection = mock_collection
        
        result = db_service.delete_collection("current_collection")
        
        assert result is True
        assert db_service.collection is None
        mock_client.delete_collection.assert_called_once_with(name="current_collection")
    
    @patch('app.services.database.chromadb.PersistentClient')
    def test_delete_collection_failure(self, mock_persistent_client, db_service):
        """测试删除集合失败"""
        # 设置模拟客户端
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_persistent_client.return_value = mock_client
        mock_client.delete_collection.side_effect = Exception("删除失败")
        
        with pytest.raises(DatabaseError) as exc_info:
            db_service.delete_collection("test_collection")
        
        assert "删除集合失败" in str(exc_info.value)
    
    @patch.object(ChromaDBService, 'delete_collection')
    @patch.object(ChromaDBService, 'create_or_get_collection')
    def test_reset_collection(self, mock_create, mock_delete, db_service):
        """测试重置集合"""
        mock_collection = Mock(spec=Collection)
        mock_create.return_value = mock_collection
        
        result = db_service.reset_collection("test_collection")
        
        assert result == mock_collection
        mock_delete.assert_called_once_with("test_collection")
        mock_create.assert_called_once_with("test_collection")
    
    def test_is_connected_false(self, db_service):
        """测试未连接状态"""
        assert db_service.is_connected() is False
    
    def test_is_connected_true(self, db_service):
        """测试已连接状态"""
        mock_client = Mock(spec=chromadb.ClientAPI)
        db_service.client = mock_client
        db_service._is_connected = True
        
        assert db_service.is_connected() is True
    
    def test_get_database_info_disconnected(self, db_service):
        """测试获取数据库信息（未连接状态）"""
        info = db_service.get_database_info()
        
        assert "database_path" in info
        assert info["is_connected"] is False
        assert info["default_collection"] == db_service.settings.COLLECTION_NAME
        assert info["current_collection"] is None
    
    @patch.object(ChromaDBService, 'list_collections')
    def test_get_database_info_connected(self, mock_list_collections, db_service):
        """测试获取数据库信息（已连接状态）"""
        # 设置连接状态
        mock_client = Mock(spec=chromadb.ClientAPI)
        mock_collection = Mock(spec=Collection)
        mock_collection.name = "current_collection"
        
        db_service.client = mock_client
        db_service.collection = mock_collection
        db_service._is_connected = True
        
        mock_list_collections.return_value = [{"name": "test"}]
        
        info = db_service.get_database_info()
        
        assert info["is_connected"] is True
        assert info["current_collection"] == "current_collection"
        assert info["total_collections"] == 1
        assert len(info["collections"]) == 1
    
    @patch.object(ChromaDBService, 'connect')
    @patch.object(ChromaDBService, 'create_or_get_collection')
    @patch.object(ChromaDBService, 'is_connected')
    def test_health_check_healthy(self, mock_is_connected, mock_create_collection, mock_connect, db_service, temp_db_dir):
        """测试健康检查（健康状态）"""
        # 创建数据库目录
        Path(temp_db_dir).mkdir(exist_ok=True)
        
        # 设置模拟
        mock_is_connected.return_value = True
        mock_create_collection.return_value = Mock(spec=Collection)
        
        result = db_service.health_check()
        
        assert result["status"] == "healthy"
        assert result["database_path_exists"] is True
        assert result["connection_status"] is True
        assert result["default_collection_exists"] is True
        assert result["error"] is None
    
    @patch.object(ChromaDBService, 'connect')
    def test_health_check_connection_failure(self, mock_connect, db_service):
        """测试健康检查（连接失败）"""
        mock_connect.side_effect = Exception("连接失败")
        
        result = db_service.health_check()
        
        assert result["status"] == "error"
        assert "连接失败" in result["error"]
    
    @patch.object(ChromaDBService, 'connect')
    @patch.object(ChromaDBService, 'create_or_get_collection')
    @patch.object(ChromaDBService, 'is_connected')
    def test_health_check_collection_failure(self, mock_is_connected, mock_create_collection, mock_connect, db_service, temp_db_dir):
        """测试健康检查（集合操作失败）"""
        # 创建数据库目录
        Path(temp_db_dir).mkdir(exist_ok=True)
        
        # 设置模拟
        mock_is_connected.return_value = True
        mock_create_collection.side_effect = Exception("集合创建失败")
        
        result = db_service.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["database_path_exists"] is True
        assert result["connection_status"] is True
        assert result["default_collection_exists"] is False
        assert "集合检查失败" in result["error"]