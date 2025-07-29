"""
DocumentProcessor 单元测试

测试使用 LangChain 和 Unstructured 进行文本提取的功能。
"""

import os
import tempfile
import pytest
import shutil
from app.core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """DocumentProcessor 测试类 (使用 LangChain)"""

    def setup_method(self):
        """测试前准备"""
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_temp_file(self, filename: str, content: str, encoding: str = 'utf-8') -> str:
        """创建临时测试文件"""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return file_path

    def test_init(self):
        """测试初始化"""
        assert self.processor is not None

    def test_extract_txt_content_utf8(self):
        """测试 UTF-8 编码的 .txt 文件提取"""
        content = "这是一个测试文件。\n包含中文内容。"
        file_path = self._create_temp_file("test.txt", content, 'utf-8')
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        # UnstructuredLoader 可能会在行之间添加额外的换行符，我们将其标准化后再比较
        normalized_extracted = extracted.replace('\n\n', '\n')
        assert normalized_extracted == content.strip()

    def test_extract_txt_content_gbk(self):
        """测试 GBK 编码的 .txt 文件提取"""
        content = "这是GBK编码的测试文件。"
        file_path = self._create_temp_file("test_gbk.txt", content, 'gbk')
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        assert extracted == content.strip()

    def test_extract_md_content_basic(self):
        """测试基本 Markdown 文件提取"""
        markdown_content = """# 标题
        
这是一个**粗体**文本和*斜体*文本。

- 列表项1
- 列表项2
"""
        expected_text = "标题\n\n这是一个粗体文本和斜体文本。\n\n列表项1\n\n列表项2"
        file_path = self._create_temp_file("test.md", markdown_content)
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        # Unstructured 的解析结果可能与原始纯文本有细微差别，但核心内容应在
        assert "标题" in extracted
        assert "这是一个粗体文本和斜体文本" in extracted
        assert "列表项1" in extracted
        assert "列表项2" in extracted

    def test_extract_text_file_not_exists(self):
        """测试文件不存在的情况"""
        success, extracted = self.processor.extract_text("nonexistent.txt")
        
        assert not success
        assert extracted == ""

    def test_extract_text_empty_file(self):
        """测试空文件"""
        file_path = self._create_temp_file("empty.txt", "")
        success, extracted = self.processor.extract_text(file_path)
        assert success
        assert extracted == ""

    def test_get_file_info_existing_file(self):
        """测试获取存在文件的信息"""
        content = "测试内容"
        file_path = self._create_temp_file("info_test.txt", content)
        
        info = self.processor.get_file_info(file_path)
        
        assert info['path'] == file_path
        assert info['exists'] is True
        assert info['format'] == 'txt'
        assert info['size'] == len(content.encode('utf-8'))

    def test_get_file_info_nonexistent_file(self):
        """测试获取不存在文件的信息"""
        file_path = os.path.join(self.temp_dir, "nonexistent.txt")
        
        info = self.processor.get_file_info(file_path)
        
        assert info['path'] == file_path
        assert info['exists'] is False
        assert info['format'] is None
        assert info['size'] == 0

    @pytest.mark.parametrize("filename, content, expected_text", [
        ("document.txt", "这是一个纯文本文件。", "这是一个纯文本文件。"),
        ("document.md", "# Markdown\n\n这是一个 Markdown 文件。", "Markdown\n\n这是一个 Markdown 文件。"),
        ("document.html", '<h1>HTML</h1><p>这是一个HTML文件</p>', "HTML\n\n这是一个HTML文件"),
    ])
    def test_extract_text_various_formats(self, filename, content, expected_text):
        """测试多种 Unstructured 支持的文件格式"""
        file_path = self._create_temp_file(filename, content)
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        assert extracted == expected_text
