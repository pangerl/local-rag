"""
DocumentProcessor 单元测试

测试文档格式验证和文本提取功能，
包括 .txt 和 .md 格式文件的处理。
"""

import os
import tempfile
import pytest
from pathlib import Path
from app.core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """DocumentProcessor 测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
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
        processor = DocumentProcessor()
        assert processor is not None
        assert processor.SUPPORTED_FORMATS == {'.txt', '.md'}
    
    def test_validate_file_format_supported(self):
        """测试支持的文件格式验证"""
        assert self.processor.validate_file_format("test.txt")
        assert self.processor.validate_file_format("test.md")
        assert self.processor.validate_file_format("TEST.TXT")  # 大小写不敏感
        assert self.processor.validate_file_format("TEST.MD")
        assert self.processor.validate_file_format("/path/to/file.txt")
        assert self.processor.validate_file_format("file.name.with.dots.md")
    
    def test_validate_file_format_unsupported(self):
        """测试不支持的文件格式"""
        assert not self.processor.validate_file_format("test.pdf")
        assert not self.processor.validate_file_format("test.docx")
        assert not self.processor.validate_file_format("test.html")
        assert not self.processor.validate_file_format("test")  # 无扩展名
        assert not self.processor.validate_file_format("")      # 空字符串
        assert not self.processor.validate_file_format(None)    # None
    
    def test_detect_file_format(self):
        """测试文件格式检测"""
        assert self.processor.detect_file_format("test.txt") == "txt"
        assert self.processor.detect_file_format("test.md") == "md"
        assert self.processor.detect_file_format("TEST.TXT") == "txt"
        assert self.processor.detect_file_format("TEST.MD") == "md"
        
        # 不支持的格式应返回 None
        assert self.processor.detect_file_format("test.pdf") is None
        assert self.processor.detect_file_format("test") is None
        assert self.processor.detect_file_format("") is None
    
    def test_extract_txt_content_utf8(self):
        """测试 UTF-8 编码的 .txt 文件提取"""
        content = "这是一个测试文件。\n包含中文内容。"
        file_path = self._create_temp_file("test.txt", content, 'utf-8')
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        assert extracted == content.strip()
    
    def test_extract_txt_content_gbk(self):
        """测试 GBK 编码的 .txt 文件提取"""
        content = "这是GBK编码的测试文件。"
        file_path = os.path.join(self.temp_dir, "test_gbk.txt")
        
        # 创建 GBK 编码文件
        with open(file_path, 'w', encoding='gbk') as f:
            f.write(content)
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        assert extracted == content.strip()
    
    def test_extract_txt_content_empty_file(self):
        """测试空 .txt 文件"""
        file_path = self._create_temp_file("empty.txt", "")
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        assert extracted == ""
    
    def test_extract_txt_content_whitespace_only(self):
        """测试只包含空白字符的 .txt 文件"""
        content = "   \n\t  \n  "
        file_path = self._create_temp_file("whitespace.txt", content)
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        assert extracted == ""  # strip() 后应为空
    
    def test_extract_md_content_basic(self):
        """测试基本 Markdown 文件提取"""
        markdown_content = """# 标题
        
这是一个**粗体**文本和*斜体*文本。

## 子标题

- 列表项1
- 列表项2

> 这是引用文本

`代码片段`

[链接文本](http://example.com)
"""
        
        file_path = self._create_temp_file("test.md", markdown_content)
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        assert "标题" in extracted
        assert "粗体" in extracted
        assert "斜体" in extracted
        assert "子标题" in extracted
        assert "列表项1" in extracted
        assert "引用文本" in extracted
        assert "代码片段" in extracted
        assert "链接文本" in extracted
        
        # 确保 Markdown 标记被移除
        assert "#" not in extracted
        assert "**" not in extracted
        assert "*" not in extracted
        assert ">" not in extracted
        assert "`" not in extracted
        assert "[" not in extracted
        assert "]" not in extracted
        assert "(" not in extracted
        assert ")" not in extracted
    
    def test_extract_md_content_code_blocks(self):
        """测试包含代码块的 Markdown 文件"""
        markdown_content = """# 代码示例

这是普通文本。

```python
def hello():
    print("Hello, World!")
```

这是更多文本。
"""
        
        file_path = self._create_temp_file("code.md", markdown_content)
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        assert "代码示例" in extracted
        assert "这是普通文本" in extracted
        assert "这是更多文本" in extracted
        
        # 代码块应该被移除
        assert "```" not in extracted
        assert "def hello" not in extracted
        assert "print" not in extracted
    
    def test_extract_md_content_complex(self):
        """测试复杂 Markdown 文件"""
        markdown_content = """# 主标题

## 介绍

这是一个**重要**的文档，包含*多种*格式。

### 列表示例

1. 第一项
2. 第二项
   - 子项A
   - 子项B

### 链接和图片

访问 [官网](https://example.com) 获取更多信息。

![图片描述](image.png)

### 引用

> 这是一段重要的引用文字。
> 它可能包含多行内容。

### 代码

行内代码：`print("hello")`

代码块：
```javascript
function test() {
    return "test";
}
```

### 表格

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |

---

~~删除的文本~~

**结束**
"""
        
        file_path = self._create_temp_file("complex.md", markdown_content)
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert success
        
        # 检查内容是否保留
        assert "主标题" in extracted
        assert "介绍" in extracted
        assert "重要" in extracted
        assert "多种" in extracted
        assert "第一项" in extracted
        assert "子项A" in extracted
        assert "官网" in extracted
        assert "图片描述" in extracted
        assert "重要的引用文字" in extracted
        assert "删除的文本" in extracted
        assert "结束" in extracted
        
        # 检查标记是否被移除
        assert "#" not in extracted
        assert "**" not in extracted
        assert "*" not in extracted
        assert "[" not in extracted
        assert "]" not in extracted
        assert ">" not in extracted
        assert "`" not in extracted
        assert "```" not in extracted
        assert "|" not in extracted
        assert "---" not in extracted
        assert "~~" not in extracted
    
    def test_extract_text_file_not_exists(self):
        """测试文件不存在的情况"""
        success, extracted = self.processor.extract_text("nonexistent.txt")
        
        assert not success
        assert extracted == ""
    
    def test_extract_text_unsupported_format(self):
        """测试不支持的文件格式"""
        # 创建一个 .pdf 文件（实际内容是文本）
        file_path = self._create_temp_file("test.pdf", "这是PDF内容")
        
        success, extracted = self.processor.extract_text(file_path)
        
        assert not success
        assert extracted == ""
    
    def test_remove_markdown_syntax_headers(self):
        """测试移除标题标记"""
        markdown = "# 一级标题\n## 二级标题\n### 三级标题"
        result = self.processor._remove_markdown_syntax(markdown)
        
        assert "一级标题" in result
        assert "二级标题" in result
        assert "三级标题" in result
        assert "#" not in result
    
    def test_remove_markdown_syntax_emphasis(self):
        """测试移除强调标记"""
        markdown = "这是**粗体**和*斜体*以及__粗体__和_斜体_文本"
        result = self.processor._remove_markdown_syntax(markdown)
        
        assert "粗体" in result
        assert "斜体" in result
        assert "**" not in result
        assert "*" not in result
        assert "__" not in result
        assert "_" not in result
    
    def test_remove_markdown_syntax_links(self):
        """测试移除链接标记"""
        markdown = "访问[官网](https://example.com)和![图片](image.png)"
        result = self.processor._remove_markdown_syntax(markdown)
        
        assert "官网" in result
        assert "图片" in result
        assert "[" not in result
        assert "]" not in result
        assert "(" not in result
        assert ")" not in result
        assert "https://example.com" not in result
    
    def test_remove_markdown_syntax_lists(self):
        """测试移除列表标记"""
        markdown = """- 项目1
* 项目2
+ 项目3
1. 编号项目1
2. 编号项目2"""
        result = self.processor._remove_markdown_syntax(markdown)
        
        assert "项目1" in result
        assert "项目2" in result
        assert "项目3" in result
        assert "编号项目1" in result
        assert "编号项目2" in result
        assert "-" not in result
        assert "*" not in result
        assert "+" not in result
        assert "1." not in result
        assert "2." not in result
    
    def test_get_file_info_existing_file(self):
        """测试获取存在文件的信息"""
        content = "测试内容"
        file_path = self._create_temp_file("info_test.txt", content)
        
        info = self.processor.get_file_info(file_path)
        
        assert info['path'] == file_path
        assert info['exists'] is True
        assert info['format'] == 'txt'
        assert info['size'] > 0
        assert info['supported'] is True
    
    def test_get_file_info_nonexistent_file(self):
        """测试获取不存在文件的信息"""
        file_path = os.path.join(self.temp_dir, "nonexistent.txt")
        
        info = self.processor.get_file_info(file_path)
        
        assert info['path'] == file_path
        assert info['exists'] is False
        assert info['format'] is None
        assert info['size'] == 0
        assert info['supported'] is False
    
    def test_get_file_info_unsupported_format(self):
        """测试获取不支持格式文件的信息"""
        file_path = self._create_temp_file("test.pdf", "内容")
        
        info = self.processor.get_file_info(file_path)
        
        assert info['path'] == file_path
        assert info['exists'] is True
        assert info['format'] is None
        assert info['size'] > 0
        assert info['supported'] is False