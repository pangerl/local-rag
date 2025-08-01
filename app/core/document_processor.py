"""
文档处理模块 (文档加载器)

使用 LangChain 和 Unstructured 将文件加载为 Document 对象，
支持多种文档格式。
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from unstructured.partition.auto import partition
from unstructured.documents.elements import Element
from langchain_core.documents import Document
from app.core.config import settings

logger = logging.getLogger(__name__)


def _elements_to_docs(elements: List[Element], file_path: str) -> List[Document]:
    """将 Unstructured 的 Element 列表转换为 LangChain 的 Document 列表"""
    documents = []
    for element in elements:
        doc = Document(
            page_content=str(element),
            metadata={
                'source': file_path,
                'category': element.category,
                **element.metadata.to_dict()
            }
        )
        documents.append(doc)
    return documents


class DocumentProcessor:
    """
    文档加载器

    使用 LangChain 的 UnstructuredLoader 将文件加载为 Document 对象列表，
    并自动添加文件源作为元数据。
    """

    def __init__(self):
        """初始化文档加载器"""
        self.SUPPORTED_FORMATS = set(settings.SUPPORTED_FORMATS)
        logger.info("DocumentProcessor (Loader) 初始化完成 (使用 LangChain/Unstructured)")

    def load(self, file_path: str) -> List[Document]:
        """
        从文件加载并返回 Document 对象列表

        Args:
            file_path: 文件路径

        Returns:
            包含加载内容的 Document 对象列表

        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果文件为空或提取失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if os.path.getsize(file_path) == 0:
            logger.warning(f"文件为空: {file_path}")
            return []

        try:
            # 直接使用 unstructured.partition.auto 进行文档解析
            # 这种方式更底层，效率更高，并能避免 LangChain Loader 的一些问题
            elements = partition(filename=file_path)
            documents = _elements_to_docs(elements, file_path)

            logger.info(f"成功加载文件: {file_path}, 生成 {len(documents)} 个 Document")
            return documents

        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {e}")
            raise ValueError(f"无法加载文件: {file_path}") from e

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件基本信息

        Args:
            file_path: 文件路径

        Returns:
            包含文件信息的字典
        """
        info: Dict[str, Any] = {
            'path': file_path,
            'exists': False,
            'format': None,
            'size': 0,
            'is_supported': False
        }

        try:
            path = Path(file_path)
            if path.exists():
                info['exists'] = True
                info['size'] = path.stat().st_size
                file_format = path.suffix.lower()
                info['format'] = file_format.lstrip('.')
                if file_format in self.SUPPORTED_FORMATS:
                    info['is_supported'] = True

        except Exception as e:
            logger.error(f"获取文件信息失败: {file_path}, 错误: {e}")

        return info
