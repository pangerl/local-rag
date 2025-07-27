"""
文本分片模块

使用 jieba 分词实现中文优化的文本分片策略，
基于词元数量而非字符数量进行滑动窗口分片。
"""

import jieba
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class JiebaChunker:
    """
    基于 jieba 分词的中文文本分片器
    
    使用滑动窗口技术，基于词元数量进行文本分片，
    确保分片在语义上连贯且大小可控。
    """
    
    def __init__(self):
        """初始化 jieba 分词器"""
        # 设置 jieba 日志级别，避免过多输出
        jieba.setLogLevel(logging.INFO)
        logger.info("JiebaChunker 初始化完成")
    
    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        使用 jieba 分词和滑动窗口进行文本分片
        
        Args:
            text: 输入文本
            chunk_size: 每个分片的词元数量
            chunk_overlap: 相邻分片的重叠词元数量
            
        Returns:
            分片后的文本列表
            
        Raises:
            ValueError: 当参数无效时抛出异常
        """
        if not text or not text.strip():
            logger.warning("输入文本为空")
            return []
            
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
            
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负数")
            
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        
        # 使用 jieba 进行分词
        tokens = self._tokenize(text)
        
        if len(tokens) <= chunk_size:
            # 如果文本长度不超过分片大小，直接返回原文本
            return [text.strip()]
        
        # 使用滑动窗口进行分片
        token_chunks = self._sliding_window(tokens, chunk_size, chunk_overlap)
        
        # 将词元块重组为文本
        text_chunks = self._reconstruct_text(token_chunks)
        
        logger.info(f"文本分片完成: 原始词元数={len(tokens)}, 分片数={len(text_chunks)}")
        return text_chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """
        使用 jieba 进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的词元列表
        """
        # 使用 jieba 精确模式进行分词
        tokens = list(jieba.cut(text, cut_all=False))
        
        # 过滤空白词元
        tokens = [token for token in tokens if token.strip()]
        
        logger.debug(f"分词完成: 词元数={len(tokens)}")
        return tokens
    
    def _sliding_window(self, tokens: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
        """
        实现滑动窗口分片逻辑
        
        Args:
            tokens: 词元列表
            chunk_size: 每个分片的词元数量
            overlap: 相邻分片的重叠词元数量
            
        Returns:
            分片后的词元块列表
        """
        if len(tokens) <= chunk_size:
            return [tokens]
        
        chunks = []
        step = chunk_size - overlap  # 滑动步长
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk = tokens[start:end]
            chunks.append(chunk)
            
            # 如果已经到达末尾，停止分片
            if end >= len(tokens):
                break
                
            start += step
        
        logger.debug(f"滑动窗口分片完成: 分片数={len(chunks)}, 步长={step}")
        return chunks
    
    def _reconstruct_text(self, token_chunks: List[List[str]]) -> List[str]:
        """
        将词元块重组为文本
        
        Args:
            token_chunks: 词元块列表
            
        Returns:
            重组后的文本列表
        """
        text_chunks = []
        
        for tokens in token_chunks:
            # 将词元连接成文本，保持中文文本的可读性
            text = self._join_tokens(tokens)
            if text.strip():  # 只添加非空文本
                text_chunks.append(text.strip())
        
        return text_chunks
    
    def _join_tokens(self, tokens: List[str]) -> str:
        """
        智能连接词元为文本
        
        对于中文文本，需要正确处理词元间的连接，
        确保标点符号和中英文混合文本的正确显示。
        
        Args:
            tokens: 词元列表
            
        Returns:
            连接后的文本
        """
        if not tokens:
            return ""
        
        result = []
        
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            else:
                prev_token = tokens[i-1]
                
                # 判断是否需要添加空格
                if self._needs_space(prev_token, token):
                    result.append(" ")
                
                result.append(token)
        
        return "".join(result)
    
    def _needs_space(self, prev_token: str, current_token: str) -> bool:
        """
        判断两个词元之间是否需要空格
        
        Args:
            prev_token: 前一个词元
            current_token: 当前词元
            
        Returns:
            是否需要空格
        """
        # 标点符号前后不需要空格
        punctuation = "，。！？；：""''（）【】《》、"
        
        if prev_token in punctuation or current_token in punctuation:
            return False
        
        # 如果前一个词元以标点结尾，不需要空格
        if prev_token and prev_token[-1] in punctuation:
            return False
            
        # 如果当前词元以标点开头，不需要空格
        if current_token and current_token[0] in punctuation:
            return False
        
        # 中文字符之间通常不需要空格
        if self._is_chinese_char(prev_token[-1]) and self._is_chinese_char(current_token[0]):
            return False
        
        # 中文和英文之间需要空格
        if self._is_chinese_char(prev_token[-1]) and current_token[0].isalpha():
            return True
            
        if prev_token[-1].isalpha() and self._is_chinese_char(current_token[0]):
            return True
        
        # 英文单词之间需要空格
        if prev_token.isalnum() and current_token.isalnum():
            return True
        
        # 数字和单位之间不需要空格
        if prev_token.isdigit() and len(current_token) == 1:
            return False
        
        return False
    
    def _is_chinese_char(self, char: str) -> bool:
        """
        判断字符是否为中文字符
        
        Args:
            char: 单个字符
            
        Returns:
            是否为中文字符
        """
        return '\u4e00' <= char <= '\u9fff'