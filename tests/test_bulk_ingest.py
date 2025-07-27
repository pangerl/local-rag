"""
批量导入脚本测试
测试 bulk_ingest.py 脚本的功能
"""

import pytest
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch


class TestBulkIngestScript:
    """批量导入脚本测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_documents(self, temp_dir):
        """创建测试文档"""
        docs_dir = Path(temp_dir) / "documents"
        docs_dir.mkdir()
        
        # 创建测试文档
        (docs_dir / "doc1.txt").write_text("这是第一个测试文档的内容。包含多个句子用于测试分片功能。", encoding='utf-8')
        (docs_dir / "doc2.md").write_text("# 测试 Markdown 文档\n\n这是 **Markdown** 格式的测试内容。", encoding='utf-8')
        
        # 创建子目录
        sub_dir = docs_dir / "subdocs"
        sub_dir.mkdir()
        (sub_dir / "doc3.txt").write_text("这是子目录中的测试文档。", encoding='utf-8')
        
        # 创建不支持的格式文件
        (docs_dir / "unsupported.pdf").write_text("PDF content", encoding='utf-8')
        
        return docs_dir
    
    def test_script_help(self):
        """测试脚本帮助信息"""
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "批量文档导入工具" in result.stdout
        assert "--path" in result.stdout
        assert "--chunk-size" in result.stdout
        assert "--chunk-overlap" in result.stdout
    
    def test_script_missing_path(self):
        """测试缺少路径参数"""
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "required" in result.stderr or "error" in result.stderr
    
    def test_script_nonexistent_path(self):
        """测试不存在的路径"""
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", "/nonexistent/path"
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "路径不存在" in result.stdout
    
    def test_script_invalid_chunk_parameters(self):
        """测试无效的分片参数"""
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", ".",
            "--chunk-size", "100",
            "--chunk-overlap", "100"
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "chunk-overlap 必须小于 chunk-size" in result.stdout
    
    def test_script_dry_run(self, test_documents):
        """测试 dry-run 模式"""
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(test_documents),
            "--dry-run"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "找到" in result.stdout and "个文档文件" in result.stdout
        assert "Dry-run 模式" in result.stdout
        assert "doc1.txt" in result.stdout
        assert "doc2.md" in result.stdout
        assert "doc3.txt" in result.stdout
    
    def test_script_dry_run_no_recursive(self, test_documents):
        """测试非递归 dry-run 模式"""
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(test_documents),
            "--no-recursive",
            "--dry-run"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "doc1.txt" in result.stdout
        assert "doc2.md" in result.stdout
        # 子目录中的文档不应该被找到
        assert "doc3.txt" not in result.stdout
    
    def test_script_single_file_dry_run(self, test_documents):
        """测试单文件 dry-run 模式"""
        single_file = test_documents / "doc1.txt"
        
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(single_file),
            "--dry-run"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "找到 1 个文档文件" in result.stdout
        assert "doc1.txt" in result.stdout
    
    def test_script_unsupported_file_dry_run(self, test_documents):
        """测试不支持的文件格式"""
        unsupported_file = test_documents / "unsupported.pdf"
        
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(unsupported_file),
            "--dry-run"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "未找到支持的文档文件" in result.stdout
    
    def test_script_empty_directory_dry_run(self, temp_dir):
        """测试空目录"""
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(empty_dir),
            "--dry-run"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "未找到支持的文档文件" in result.stdout
    
    def test_script_custom_parameters_dry_run(self, test_documents):
        """测试自定义参数"""
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(test_documents),
            "--chunk-size", "200",
            "--chunk-overlap", "30",
            "--log-level", "DEBUG",
            "--dry-run"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "找到" in result.stdout and "个文档文件" in result.stdout
    
    @patch('scripts.bulk_ingest.DocumentService')
    @patch('scripts.bulk_ingest.ChromaDBService')
    @patch('scripts.bulk_ingest.ModelLoader')
    def test_script_processing_success(self, mock_model_loader, mock_db_service, 
                                     mock_document_service, test_documents):
        """测试成功处理文档（使用模拟服务）"""
        # 设置模拟服务
        mock_service_instance = Mock()
        mock_service_instance.health_check.return_value = {"status": "healthy"}
        mock_service_instance.process_document.return_value = {
            "status": "success",
            "chunks_created": 3,
            "chunks_stored": 3,
            "processing_time": 1.0
        }
        mock_document_service.return_value = mock_service_instance
        
        # 只处理一个文件以简化测试
        single_file = test_documents / "doc1.txt"
        
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(single_file),
            "--chunk-size", "300",
            "--chunk-overlap", "50"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "批量处理完成" in result.stdout
        assert "成功处理: 1" in result.stdout
        assert "处理失败: 0" in result.stdout
    
    @patch('scripts.bulk_ingest.DocumentService')
    @patch('scripts.bulk_ingest.ChromaDBService')
    @patch('scripts.bulk_ingest.ModelLoader')
    def test_script_processing_failure(self, mock_model_loader, mock_db_service, 
                                     mock_document_service, test_documents):
        """测试处理失败的情况"""
        # 设置模拟服务
        mock_service_instance = Mock()
        mock_service_instance.health_check.return_value = {"status": "healthy"}
        mock_service_instance.process_document.side_effect = Exception("处理失败")
        mock_document_service.return_value = mock_service_instance
        
        single_file = test_documents / "doc1.txt"
        
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(single_file)
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "处理失败: 1" in result.stdout
    
    @patch('scripts.bulk_ingest.DocumentService')
    @patch('scripts.bulk_ingest.ChromaDBService')
    @patch('scripts.bulk_ingest.ModelLoader')
    def test_script_health_check_failure(self, mock_model_loader, mock_db_service, 
                                        mock_document_service, test_documents):
        """测试健康检查失败"""
        # 设置模拟服务
        mock_service_instance = Mock()
        mock_service_instance.health_check.return_value = {"status": "unhealthy"}
        mock_document_service.return_value = mock_service_instance
        
        single_file = test_documents / "doc1.txt"
        
        result = subprocess.run([
            sys.executable, "scripts/bulk_ingest.py",
            "--path", str(single_file)
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "系统健康检查失败" in result.stdout


class TestBulkIngestFunctions:
    """批量导入脚本函数测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_find_documents_single_file(self, temp_dir):
        """测试查找单个文件"""
        from scripts.bulk_ingest import find_documents
        
        # 创建测试文件
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content", encoding='utf-8')
        
        documents = find_documents(test_file)
        
        assert len(documents) == 1
        assert documents[0] == test_file
    
    def test_find_documents_unsupported_file(self, temp_dir):
        """测试不支持的文件格式"""
        from scripts.bulk_ingest import find_documents
        
        # 创建不支持的文件
        test_file = Path(temp_dir) / "test.pdf"
        test_file.write_text("test content", encoding='utf-8')
        
        documents = find_documents(test_file)
        
        assert len(documents) == 0
    
    def test_find_documents_directory_recursive(self, temp_dir):
        """测试递归搜索目录"""
        from scripts.bulk_ingest import find_documents
        
        # 创建目录结构
        docs_dir = Path(temp_dir)
        (docs_dir / "doc1.txt").write_text("content1", encoding='utf-8')
        (docs_dir / "doc2.md").write_text("content2", encoding='utf-8')
        
        sub_dir = docs_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "doc3.txt").write_text("content3", encoding='utf-8')
        
        documents = find_documents(docs_dir, recursive=True)
        
        assert len(documents) == 3
        assert any(doc.name == "doc1.txt" for doc in documents)
        assert any(doc.name == "doc2.md" for doc in documents)
        assert any(doc.name == "doc3.txt" for doc in documents)
    
    def test_find_documents_directory_non_recursive(self, temp_dir):
        """测试非递归搜索目录"""
        from scripts.bulk_ingest import find_documents
        
        # 创建目录结构
        docs_dir = Path(temp_dir)
        (docs_dir / "doc1.txt").write_text("content1", encoding='utf-8')
        (docs_dir / "doc2.md").write_text("content2", encoding='utf-8')
        
        sub_dir = docs_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "doc3.txt").write_text("content3", encoding='utf-8')
        
        documents = find_documents(docs_dir, recursive=False)
        
        assert len(documents) == 2
        assert any(doc.name == "doc1.txt" for doc in documents)
        assert any(doc.name == "doc2.md" for doc in documents)
        assert not any(doc.name == "doc3.txt" for doc in documents)
    
    def test_find_documents_nonexistent_path(self):
        """测试不存在的路径"""
        from scripts.bulk_ingest import find_documents
        
        nonexistent_path = Path("/nonexistent/path")
        documents = find_documents(nonexistent_path)
        
        assert len(documents) == 0
    
    def test_process_documents_success(self):
        """测试成功处理文档"""
        from scripts.bulk_ingest import process_documents
        
        # 创建模拟文档服务
        mock_service = Mock()
        mock_service.process_document.return_value = {
            "status": "success",
            "chunks_created": 5,
            "chunks_stored": 5,
            "processing_time": 1.5
        }
        
        # 创建模拟文档列表
        documents = [Path("doc1.txt"), Path("doc2.txt")]
        
        results = process_documents(mock_service, documents, 300, 50)
        
        assert results["total_documents"] == 2
        assert results["successful_documents"] == 2
        assert results["failed_documents"] == 0
        assert results["total_chunks_created"] == 10
        assert results["total_chunks_stored"] == 10
        assert len(results["processing_results"]) == 2
        assert len(results["errors"]) == 0
    
    def test_process_documents_with_failures(self):
        """测试处理文档时的失败情况"""
        from scripts.bulk_ingest import process_documents
        from app.core.exceptions import DocumentProcessError
        
        # 创建模拟文档服务
        mock_service = Mock()
        mock_service.process_document.side_effect = [
            {
                "status": "success",
                "chunks_created": 3,
                "chunks_stored": 3,
                "processing_time": 1.0
            },
            DocumentProcessError("处理失败"),
            {
                "status": "success",
                "chunks_created": 2,
                "chunks_stored": 2,
                "processing_time": 0.8
            }
        ]
        
        documents = [Path("doc1.txt"), Path("doc2.txt"), Path("doc3.txt")]
        
        results = process_documents(mock_service, documents, 300, 50)
        
        assert results["total_documents"] == 3
        assert results["successful_documents"] == 2
        assert results["failed_documents"] == 1
        assert results["total_chunks_created"] == 5
        assert results["total_chunks_stored"] == 5
        assert len(results["processing_results"]) == 2
        assert len(results["errors"]) == 1
        assert results["errors"][0]["error_type"] == "DocumentProcessError"
    
    def test_process_documents_system_error(self):
        """测试系统级错误"""
        from scripts.bulk_ingest import process_documents
        from app.core.exceptions import DatabaseError
        
        # 创建模拟文档服务
        mock_service = Mock()
        mock_service.process_document.side_effect = [
            {
                "status": "success",
                "chunks_created": 3,
                "chunks_stored": 3,
                "processing_time": 1.0
            },
            DatabaseError("数据库连接失败")  # 系统级错误，应该停止处理
        ]
        
        documents = [Path("doc1.txt"), Path("doc2.txt"), Path("doc3.txt")]
        
        results = process_documents(mock_service, documents, 300, 50)
        
        assert results["total_documents"] == 3
        assert results["successful_documents"] == 1
        assert results["failed_documents"] == 1
        # 第三个文档不应该被处理，因为遇到了系统级错误
        assert len(results["processing_results"]) == 1
        assert len(results["errors"]) == 1
        assert results["errors"][0]["error_type"] == "DatabaseError"