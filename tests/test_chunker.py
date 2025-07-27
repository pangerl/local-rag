"""
JiebaChunker 单元测试

测试 jieba 分词和文本分片功能的正确性，
包括中文文本的分词效果和语义连贯性验证。
"""

import pytest
from app.core.chunker import JiebaChunker


class TestJiebaChunker:
    """JiebaChunker 测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.chunker = JiebaChunker()
    
    def test_init(self):
        """测试初始化"""
        chunker = JiebaChunker()
        assert chunker is not None
    
    def test_tokenize_chinese_text(self):
        """测试中文文本分词"""
        text = "我爱北京天安门，天安门上太阳升。"
        tokens = self.chunker._tokenize(text)
        
        # 验证分词结果不为空
        assert len(tokens) > 0
        
        # 验证分词结果包含预期的词汇
        assert "我" in tokens
        assert "爱" in tokens
        assert "北京" in tokens
        assert "天安门" in tokens
        
        # 验证没有空白词元
        for token in tokens:
            assert token.strip() != ""
    
    def test_tokenize_mixed_text(self):
        """测试中英文混合文本分词"""
        text = "我使用Python编程，它很好用。"
        tokens = self.chunker._tokenize(text)
        
        assert len(tokens) > 0
        assert "Python" in tokens
        assert "编程" in tokens
    
    def test_tokenize_empty_text(self):
        """测试空文本分词"""
        tokens = self.chunker._tokenize("")
        assert tokens == []
        
        tokens = self.chunker._tokenize("   ")
        assert tokens == []
    
    def test_sliding_window_basic(self):
        """测试基本滑动窗口功能"""
        tokens = ["我", "爱", "北京", "天安门", "上", "太阳", "升"]
        chunk_size = 3
        overlap = 1
        
        chunks = self.chunker._sliding_window(tokens, chunk_size, overlap)
        
        # 验证分片数量
        expected_chunks = [
            ["我", "爱", "北京"],
            ["北京", "天安门", "上"],
            ["上", "太阳", "升"]
        ]
        assert chunks == expected_chunks
    
    def test_sliding_window_no_overlap(self):
        """测试无重叠的滑动窗口"""
        tokens = ["我", "爱", "北京", "天安门", "上", "太阳"]
        chunk_size = 2
        overlap = 0
        
        chunks = self.chunker._sliding_window(tokens, chunk_size, overlap)
        
        expected_chunks = [
            ["我", "爱"],
            ["北京", "天安门"],
            ["上", "太阳"]
        ]
        assert chunks == expected_chunks
    
    def test_sliding_window_short_text(self):
        """测试短文本滑动窗口"""
        tokens = ["我", "爱"]
        chunk_size = 5
        overlap = 1
        
        chunks = self.chunker._sliding_window(tokens, chunk_size, overlap)
        
        # 短文本应该返回单个分片
        assert len(chunks) == 1
        assert chunks[0] == tokens
    
    def test_join_tokens_chinese(self):
        """测试中文词元连接"""
        tokens = ["我", "爱", "北京", "天安门"]
        result = self.chunker._join_tokens(tokens)
        
        # 中文词元之间不应该有空格
        assert result == "我爱北京天安门"
    
    def test_join_tokens_with_punctuation(self):
        """测试包含标点符号的词元连接"""
        tokens = ["我", "爱", "北京", "，", "天安门", "很", "美丽", "。"]
        result = self.chunker._join_tokens(tokens)
        
        # 标点符号前后不应该有多余空格
        assert result == "我爱北京，天安门很美丽。"
    
    def test_join_tokens_mixed_language(self):
        """测试中英文混合词元连接"""
        tokens = ["我", "使用", "Python", "编程"]
        result = self.chunker._join_tokens(tokens)
        
        # 应该正确处理中英文混合，中英文之间应该有空格
        assert "Python" in result
        assert result == "我使用 Python 编程"
    
    def test_needs_space_chinese(self):
        """测试中文字符间空格判断"""
        # 中文字符之间不需要空格
        assert not self.chunker._needs_space("我", "爱")
        assert not self.chunker._needs_space("北京", "天安门")
    
    def test_needs_space_punctuation(self):
        """测试标点符号空格判断"""
        # 标点符号前后不需要空格
        assert not self.chunker._needs_space("我", "，")
        assert not self.chunker._needs_space("。", "天安门")
    
    def test_needs_space_english(self):
        """测试英文单词空格判断"""
        # 英文单词之间需要空格
        assert self.chunker._needs_space("hello", "world")
        assert self.chunker._needs_space("Python", "programming")
    
    def test_is_chinese_char(self):
        """测试中文字符判断"""
        assert self.chunker._is_chinese_char("我")
        assert self.chunker._is_chinese_char("爱")
        assert self.chunker._is_chinese_char("中")
        
        assert not self.chunker._is_chinese_char("a")
        assert not self.chunker._is_chinese_char("1")
        assert not self.chunker._is_chinese_char("，")
    
    def test_chunk_text_basic(self):
        """测试基本文本分片功能"""
        text = "我爱北京天安门，天安门上太阳升。伟大的祖国繁荣昌盛。"
        chunk_size = 5
        chunk_overlap = 2
        
        chunks = self.chunker.chunk_text(text, chunk_size, chunk_overlap)
        
        # 验证返回结果
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(chunk.strip() for chunk in chunks)  # 所有分片都不为空
    
    def test_chunk_text_short_text(self):
        """测试短文本分片"""
        text = "我爱北京"
        chunk_size = 10
        chunk_overlap = 2
        
        chunks = self.chunker.chunk_text(text, chunk_size, chunk_overlap)
        
        # 短文本应该返回单个分片
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_empty_input(self):
        """测试空输入"""
        chunks = self.chunker.chunk_text("", 5, 1)
        assert chunks == []
        
        chunks = self.chunker.chunk_text("   ", 5, 1)
        assert chunks == []
    
    def test_chunk_text_invalid_params(self):
        """测试无效参数"""
        text = "我爱北京天安门"
        
        # chunk_size 必须大于 0
        with pytest.raises(ValueError, match="chunk_size 必须大于 0"):
            self.chunker.chunk_text(text, 0, 1)
        
        with pytest.raises(ValueError, match="chunk_size 必须大于 0"):
            self.chunker.chunk_text(text, -1, 1)
        
        # chunk_overlap 不能为负数
        with pytest.raises(ValueError, match="chunk_overlap 不能为负数"):
            self.chunker.chunk_text(text, 5, -1)
        
        # chunk_overlap 必须小于 chunk_size
        with pytest.raises(ValueError, match="chunk_overlap 必须小于 chunk_size"):
            self.chunker.chunk_text(text, 5, 5)
        
        with pytest.raises(ValueError, match="chunk_overlap 必须小于 chunk_size"):
            self.chunker.chunk_text(text, 5, 6)
    
    def test_chunk_text_semantic_coherence(self):
        """测试语义连贯性"""
        text = """
        人工智能是计算机科学的一个分支，它企图了解智能的实质，
        并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        """
        
        chunk_size = 10
        chunk_overlap = 3
        
        chunks = self.chunker.chunk_text(text, chunk_size, chunk_overlap)
        
        # 验证分片结果
        assert len(chunks) > 1  # 应该产生多个分片
        
        # 验证每个分片都是有意义的文本
        for chunk in chunks:
            assert len(chunk.strip()) > 0
            # 分片应该包含完整的词汇，而不是被截断的字符
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")
    
    def test_chunk_text_overlap_consistency(self):
        """测试重叠一致性"""
        text = "我爱北京天安门，天安门上太阳升，伟大的祖国繁荣昌盛。"
        chunk_size = 6
        chunk_overlap = 2
        
        chunks = self.chunker.chunk_text(text, chunk_size, chunk_overlap)
        
        if len(chunks) > 1:
            # 验证相邻分片之间确实有重叠内容
            # 这里我们检查是否有共同的词汇
            for i in range(len(chunks) - 1):
                current_tokens = self.chunker._tokenize(chunks[i])
                next_tokens = self.chunker._tokenize(chunks[i + 1])
                
                # 应该有一些重叠的词元
                common_tokens = set(current_tokens) & set(next_tokens)
                assert len(common_tokens) > 0, f"分片 {i} 和 {i+1} 之间没有重叠内容"
    
    def test_chunk_text_with_numbers_and_english(self):
        """测试包含数字和英文的文本分片"""
        text = "Python 3.8版本发布于2019年，包含了许多新特性。机器学习库如TensorFlow 2.0也很流行。"
        chunk_size = 8
        chunk_overlap = 2
        
        chunks = self.chunker.chunk_text(text, chunk_size, chunk_overlap)
        
        # 验证分片结果包含英文和数字
        all_text = " ".join(chunks)
        assert "Python" in all_text
        assert "3.8" in all_text
        assert "2019" in all_text
        assert "TensorFlow" in all_text
        assert "2.0" in all_text