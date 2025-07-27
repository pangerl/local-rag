"""
文档处理服务集成测试
测试 DocumentService 的完整文档处理流程集成功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.services.document_service import DocumentService
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import DocumentProcessError, UnsupportedFormatError, FileNotFoundError


class TestDocumentServiceIntegration:
    """DocumentService 集成测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_settings(self, temp_dir):
        """创建测试用的配置对象"""
        settings = Settings()
        settings.CHROMA_DB_PATH = str(Path(temp_dir) / "chroma_db")
        settings.COLLECTION_NAME = "test_document_service"
        settings.DEFAULT_CHUNK_SIZE = 100  # 较小的分片用于测试
        settings.DEFAULT_CHUNK_OVERLAP = 20
        return settings
    
    @pytest.fixture
    def db_service(self, test_settings):
        """创建 ChromaDBService 实例"""
        service = ChromaDBService(test_settings)
        yield service
        service.disconnect()
    
    @pytest.fixture
    def mock_model_loader(self):
        """创建模拟模型加载器"""
        loader = Mock(spec=ModelLoader)
        
        # 创建模拟的嵌入模型
        mock_model = Mock(spec=SentenceTransformer)
        
        def mock_encode(texts, **kwargs):
            # 返回模拟的嵌入向量（每个文本对应一个3维向量）
            embeddings = []
            for i, text in enumerate(texts):
                # 基于文本内容生成简单的向量
                vector = [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1]
                embeddings.append(vector)
            return np.array(embeddings)
        
        mock_model.encode = mock_encode
        loader.load_embedding_model.return_value = mock_model
        loader.get_model_info.return_value = {
            "models_loaded": True,
            "embedding_model_loaded": True,
            "reranker_model_loaded": True
        }
        
        return loader
    
    @pytest.fixture
    def document_service(self, test_settings, db_service, mock_model_loader):
        """创建 DocumentService 实例"""
        return DocumentService(test_settings, db_service, mock_model_loader)
    
    @pytest.fixture
    def test_txt_file(self, temp_dir):
        """创建测试用的 .txt 文件"""
        content = """这是一个测试文档的内容。
        
文档包含多个段落，用于测试文档处理服务的完整功能。
每个段落都有足够的内容来生成多个文本分片。

第二段落包含更多的文本内容，确保能够测试分片功能。
这里有更多的句子来增加文本长度。
分词器会将这些内容分解为词元，然后进行滑动窗口分片。

第三段落继续添加内容，测试批量处理功能。
文档处理服务会将整个文档分解为多个分片。
每个分片都会被向量化并存储到数据库中。"""
        
        file_path = Path(temp_dir) / "test_document.txt"
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)
    
    @pytest.fixture
    def test_md_file(self, temp_dir):
        """创建测试用的 .md 文件"""
        content = """# 测试 Markdown 文档

这是一个 **Markdown** 格式的测试文档。

## 第一节

包含 *斜体* 和 **粗体** 文本的段落。

- 列表项目 1
- 列表项目 2
- 列表项目 3

## 第二节

包含 [链接](http://example.com) 和 `代码` 的段落。

```python
# 这是代码块
def hello():
    print("Hello, World!")
```

> 这是引用文本

## 第三节

更多的文本内容用于测试 Markdown 解析功能。
确保能够正确提取纯文本内容。"""
        
        file_path = Path(temp_dir) / "test_document.md"
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)
    
    def test_process_txt_document_complete_flow(self, document_service, test_txt_file):
        """测试处理 .txt 文档的完整流程"""
        result = document_service.process_document(test_txt_file)
        
        assert result["status"] == "success"
        assert result["document_path"] == test_txt_file
        assert result["chunks_created"] > 0
        assert result["chunks_stored"] == result["chunks_created"]
        assert result["text_length"] > 0
        assert result["processing_time"] > 0
        assert result["chunk_size"] == 100
        assert result["chunk_overlap"] == 20
        
        # 验证文档已存储
        doc_info = document_service.get_document_info(test_txt_file)
        assert doc_info["file_info"]["exists"] is True
        assert doc_info["storage_info"]["is_stored"] is True
        assert doc_info["storage_info"]["total_chunks"] == result["chunks_created"]
    
    def test_process_md_document_complete_flow(self, document_service, test_md_file):
        """测试处理 .md 文档的完整流程"""
        result = document_service.process_document(test_md_file, chunk_size=80, chunk_overlap=15)
        
        assert result["status"] == "success"
        assert result["document_path"] == test_md_file
        assert result["chunks_created"] > 0
        assert result["chunks_stored"] == result["chunks_created"]
        assert result["chunk_size"] == 80
        assert result["chunk_overlap"] == 15
        
        # 验证 Markdown 内容被正确处理
        doc_info = document_service.get_document_info(test_md_file)
        chunks = doc_info["storage_info"]["chunks"]
        
        # 检查是否移除了 Markdown 标记
        for chunk in chunks:
            text = chunk["text"]
            assert "**" not in text  # 粗体标记应被移除
            assert "*" not in text   # 斜体标记应被移除
            assert "#" not in text   # 标题标记应被移除
            assert "[" not in text   # 链接标记应被移除
            assert "`" not in text   # 代码标记应被移除
    
    def test_update_document_flow(self, document_service, test_txt_file, temp_dir):
        """测试文档更新流程"""
        # 首先处理原始文档
        original_result = document_service.process_document(test_txt_file)
        original_chunks = original_result["chunks_created"]
        
        # 修改文档内容
        new_content = "这是更新后的文档内容。\n内容已经完全改变，用于测试更新功能。"
        Path(test_txt_file).write_text(new_content, encoding='utf-8')
        
        # 更新文档
        update_result = document_service.update_document(test_txt_file)
        
        assert update_result["status"] == "success"
        assert update_result["old_chunks_deleted"] == original_chunks
        assert update_result["new_chunks_stored"] > 0
        
        # 验证更新后的内容
        doc_info = document_service.get_document_info(test_txt_file)
        assert doc_info["storage_info"]["total_chunks"] == update_result["new_chunks_stored"]
        
        # 验证内容确实更新了
        chunks = doc_info["storage_info"]["chunks"]
        updated_text = " ".join([chunk["text"] for chunk in chunks])
        assert "更新后的文档内容" in updated_text
    
    def test_delete_document_flow(self, document_service, test_txt_file):
        """测试文档删除流程"""
        # 首先处理文档
        process_result = document_service.process_document(test_txt_file)
        chunks_created = process_result["chunks_created"]
        
        # 验证文档已存储
        doc_info = document_service.get_document_info(test_txt_file)
        assert doc_info["storage_info"]["is_stored"] is True
        
        # 删除文档
        delete_result = document_service.delete_document(test_txt_file)
        
        assert delete_result["status"] == "success"
        assert delete_result["chunks_deleted"] == chunks_created
        
        # 验证文档已删除
        doc_info = document_service.get_document_info(test_txt_file)
        assert doc_info["storage_info"]["is_stored"] is False
        assert doc_info["storage_info"]["total_chunks"] == 0
    
    def test_batch_process_documents_flow(self, document_service, test_txt_file, test_md_file, temp_dir):
        """测试批量处理文档流程"""
        # 创建额外的测试文件
        extra_file = Path(temp_dir) / "extra_test.txt"
        extra_file.write_text("额外的测试文档内容，用于批量处理测试。", encoding='utf-8')
        
        document_paths = [test_txt_file, test_md_file, str(extra_file)]
        
        # 批量处理
        batch_result = document_service.batch_process_documents(document_paths, chunk_size=60)
        
        assert batch_result["total_documents"] == 3
        assert batch_result["successful_documents"] == 3
        assert batch_result["failed_documents"] == 0
        assert len(batch_result["processing_results"]) == 3
        assert len(batch_result["errors"]) == 0
        
        # 验证所有文档都已处理
        for doc_path in document_paths:
            doc_info = document_service.get_document_info(doc_path)
            assert doc_info["storage_info"]["is_stored"] is True
            assert doc_info["storage_info"]["total_chunks"] > 0
    
    def test_batch_process_with_errors(self, document_service, test_txt_file, temp_dir):
        """测试批量处理包含错误的情况"""
        # 创建不存在的文件路径
        nonexistent_file = str(Path(temp_dir) / "nonexistent.txt")
        
        # 创建不支持的格式文件
        unsupported_file = Path(temp_dir) / "test.pdf"
        unsupported_file.write_text("PDF content", encoding='utf-8')
        
        document_paths = [test_txt_file, nonexistent_file, str(unsupported_file)]
        
        # 批量处理
        batch_result = document_service.batch_process_documents(document_paths)
        
        assert batch_result["total_documents"] == 3
        assert batch_result["successful_documents"] == 1
        assert batch_result["failed_documents"] == 2
        assert len(batch_result["processing_results"]) == 1
        assert len(batch_result["errors"]) == 2
        
        # 验证错误信息
        error_paths = [error["document_path"] for error in batch_result["errors"]]
        assert nonexistent_file in error_paths
        assert str(unsupported_file) in error_paths
    
    def test_processing_stats_tracking(self, document_service, test_txt_file, test_md_file):
        """测试处理统计信息跟踪"""
        # 初始统计信息
        initial_stats = document_service.get_processing_stats()
        assert initial_stats["total_documents_processed"] == 0
        assert initial_stats["total_chunks_created"] == 0
        
        # 处理第一个文档
        result1 = document_service.process_document(test_txt_file)
        stats1 = document_service.get_processing_stats()
        
        assert stats1["total_documents_processed"] == 1
        assert stats1["total_chunks_created"] == result1["chunks_created"]
        assert stats1["total_processing_time"] > 0
        assert stats1["last_processed_at"] is not None
        
        # 处理第二个文档
        result2 = document_service.process_document(test_md_file)
        stats2 = document_service.get_processing_stats()
        
        assert stats2["total_documents_processed"] == 2
        assert stats2["total_chunks_created"] == result1["chunks_created"] + result2["chunks_created"]
        assert stats2["total_processing_time"] > stats1["total_processing_time"]
    
    def test_error_handling_file_not_found(self, document_service):
        """测试文件不存在错误处理"""
        with pytest.raises(FileNotFoundError) as exc_info:
            document_service.process_document("nonexistent.txt")
        
        assert "文件不存在" in str(exc_info.value)
    
    def test_error_handling_unsupported_format(self, document_service, temp_dir):
        """测试不支持格式错误处理"""
        # 创建不支持的格式文件
        unsupported_file = Path(temp_dir) / "test.pdf"
        unsupported_file.write_text("PDF content", encoding='utf-8')
        
        with pytest.raises(UnsupportedFormatError) as exc_info:
            document_service.process_document(str(unsupported_file))
        
        assert "不支持的文件格式" in str(exc_info.value)
    
    def test_error_handling_empty_file(self, document_service, temp_dir):
        """测试空文件错误处理"""
        # 创建空文件
        empty_file = Path(temp_dir) / "empty.txt"
        empty_file.write_text("", encoding='utf-8')
        
        with pytest.raises(DocumentProcessError) as exc_info:
            document_service.process_document(str(empty_file))
        
        assert "文档内容为空" in str(exc_info.value)
    
    def test_health_check_integration(self, document_service):
        """测试健康检查集成功能"""
        # 先触发数据库连接，确保数据库路径存在
        document_service.db_service.connect()
        
        health = document_service.health_check()
        
        # 检查各个组件的状态
        assert health["components"]["database"]["status"] == "healthy"
        assert health["components"]["models"]["status"] == "healthy"
        assert health["components"]["document_processor"]["status"] == "healthy"
        assert health["components"]["chunker"]["status"] == "healthy"
        assert health["error"] is None
        
        # 整体状态应该是健康的
        assert health["status"] == "healthy"
    
    def test_custom_chunk_parameters(self, document_service, test_txt_file):
        """测试自定义分片参数"""
        # 使用自定义参数处理文档
        result = document_service.process_document(
            test_txt_file, 
            chunk_size=50, 
            chunk_overlap=10
        )
        
        assert result["status"] == "success"
        assert result["chunk_size"] == 50
        assert result["chunk_overlap"] == 10
        
        # 验证分片元数据
        doc_info = document_service.get_document_info(test_txt_file)
        chunks = doc_info["storage_info"]["chunks"]
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert "chunk_length" in metadata
            assert "created_at" in metadata
    
    def test_document_metadata_integrity(self, document_service, test_txt_file):
        """测试文档元数据完整性"""
        result = document_service.process_document(test_txt_file)
        
        # 获取文档信息
        doc_info = document_service.get_document_info(test_txt_file)
        chunks = doc_info["storage_info"]["chunks"]
        
        # 验证元数据完整性
        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"]
            
            assert metadata["document_path"] == test_txt_file
            assert metadata["chunk_index"] == i
            assert metadata["total_chunks"] == len(chunks)
            assert metadata["chunk_length"] > 0
            assert "created_at" in metadata
            assert "chunk_id" in metadata
    
    def test_concurrent_document_operations(self, document_service, temp_dir):
        """测试并发文档操作"""
        # 创建多个测试文件
        files = []
        for i in range(3):
            file_path = Path(temp_dir) / f"concurrent_test_{i}.txt"
            content = f"这是并发测试文档 {i} 的内容。\n包含足够的文本来生成多个分片。"
            file_path.write_text(content, encoding='utf-8')
            files.append(str(file_path))
        
        # 批量处理
        batch_result = document_service.batch_process_documents(files)
        
        assert batch_result["successful_documents"] == 3
        assert batch_result["failed_documents"] == 0
        
        # 验证所有文档都正确存储
        for file_path in files:
            doc_info = document_service.get_document_info(file_path)
            assert doc_info["storage_info"]["is_stored"] is True
            assert doc_info["storage_info"]["total_chunks"] > 0
        
        # 验证统计信息
        stats = document_service.get_processing_stats()
        assert stats["total_documents_processed"] == 3
        assert stats["storage_total_documents"] == 3