"""
文本分片模块

使用 langchain 实现文本分片策略，
基于字符数量进行分片。
"""

from typing import List
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextChunker:
    """
    基于 langchain 的文本分片器

    利用 RecursiveCharacterTextSplitter 实现，
    以字符为单位进行文本分割。
    """

    def __init__(self):
        """初始化分片器"""
        logger.info("TextChunker 初始化完成")

    def _get_text_splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        """创建一个配置好的 RecursiveCharacterTextSplitter 实例"""

        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负数")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "。", "！", "？", "\n", "，", "、", ""]
        )

    def split_documents(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """
        将 Document 列表分割成更小的 Document 列表

        Args:
            documents: Document 对象列表
            chunk_size: 每个分片的最大字符数量
            chunk_overlap: 相邻分片的重叠字符数量

        Returns:
            分割后的 Document 对象列表
        """
        if not documents:
            return []

        text_splitter = self._get_text_splitter(chunk_size, chunk_overlap)

        split_docs = text_splitter.split_documents(documents)

        logger.info(f"文档分割完成: 原始文档数={len(documents)}, 分割后文档数={len(split_docs)}")
        return split_docs
