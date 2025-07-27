"""
ChromaDB 数据库服务集成测试
测试 ChromaDBService 与真实 ChromaDB 的集成功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from app.core.config import Settings
from app.services.database import ChromaDBService
from app.core.exceptions import DatabaseError


class TestChromaDBServiceIntegration:
    """ChromaDBService 集成测试类"""
    
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
        settings.COLLECTION_NAME = "test_integration"
        return settings
    
    @pytest.fixture
    def db_service(self, test_settings):
        """创建 ChromaDBService 实例"""
        service = ChromaDBService(test_settings)
        yield service
        # 清理连接
        service.disconnect()
    
    def test_real_database_connection(self, db_service):
        """测试真实数据库连接"""
        # 连接数据库
        client = db_service.connect()
        assert client is not None
        assert db_service.is_connected() is True
        
        # 验证数据库目录被创建
        assert db_service.settings.chroma_db_full_path.exists()
    
    def test_real_collection_operations(self, db_service):
        """测试真实集合操作"""
        # 连接数据库
        db_service.connect()
        
        # 创建集合
        collection = db_service.create_or_get_collection("test_collection")
        assert collection is not None
        assert collection.name == "test_collection"
        
        # 获取集合信息
        info = db_service.get_collection_info("test_collection")
        assert info["name"] == "test_collection"
        assert info["count"] == 0  # 新集合应该为空
        
        # 列出集合
        collections = db_service.list_collections()
        assert len(collections) >= 1
        assert any(c["name"] == "test_collection" for c in collections)
    
    def test_collection_lifecycle(self, db_service):
        """测试集合生命周期"""
        # 连接数据库
        db_service.connect()
        
        collection_name = "lifecycle_test"
        
        # 创建集合
        collection = db_service.create_or_get_collection(collection_name)
        assert collection.name == collection_name
        
        # 验证集合存在
        collections = db_service.list_collections()
        assert any(c["name"] == collection_name for c in collections)
        
        # 删除集合
        result = db_service.delete_collection(collection_name)
        assert result is True
        
        # 验证集合已删除
        collections = db_service.list_collections()
        assert not any(c["name"] == collection_name for c in collections)
    
    def test_collection_reset(self, db_service):
        """测试集合重置功能"""
        # 连接数据库
        db_service.connect()
        
        collection_name = "reset_test"
        
        # 创建集合
        original_collection = db_service.create_or_get_collection(collection_name)
        original_id = original_collection.id
        
        # 重置集合
        new_collection = db_service.reset_collection(collection_name)
        assert new_collection.name == collection_name
        # 重置后应该是新的集合实例
        assert new_collection.id != original_id
    
    def test_health_check_integration(self, db_service):
        """测试健康检查集成功能"""
        # 执行健康检查
        health = db_service.health_check()
        
        assert health["status"] == "healthy"
        assert health["database_path_exists"] is True
        assert health["connection_status"] is True
        assert health["default_collection_exists"] is True
        assert health["error"] is None
    
    def test_database_info_integration(self, db_service):
        """测试数据库信息获取集成功能"""
        # 连接并创建一些集合
        db_service.connect()
        db_service.create_or_get_collection("info_test_1")
        db_service.create_or_get_collection("info_test_2")
        
        # 获取数据库信息
        info = db_service.get_database_info()
        
        assert info["is_connected"] is True
        assert info["total_collections"] >= 2
        assert len(info["collections"]) >= 2
        assert any(c["name"] == "info_test_1" for c in info["collections"])
        assert any(c["name"] == "info_test_2" for c in info["collections"])
    
    def test_error_handling_integration(self, db_service):
        """测试错误处理集成功能"""
        # 测试获取不存在集合的信息
        db_service.connect()
        
        with pytest.raises(DatabaseError):
            db_service.get_collection_info("nonexistent_collection")
        
        # 测试删除不存在的集合
        with pytest.raises(DatabaseError):
            db_service.delete_collection("nonexistent_collection")
    
    def test_concurrent_operations(self, db_service):
        """测试并发操作"""
        # 连接数据库
        db_service.connect()
        
        # 创建多个集合
        collection_names = [f"concurrent_test_{i}" for i in range(5)]
        collections = []
        
        for name in collection_names:
            collection = db_service.create_or_get_collection(name)
            collections.append(collection)
            assert collection.name == name
        
        # 验证所有集合都存在
        all_collections = db_service.list_collections()
        for name in collection_names:
            assert any(c["name"] == name for c in all_collections)
        
        # 清理集合
        for name in collection_names:
            db_service.delete_collection(name)