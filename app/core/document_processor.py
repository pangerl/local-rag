"""
文档处理模块

支持多种文档格式的验证和文本提取，
包括 .txt 和 .md 格式的文件处理。
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    文档处理器
    
    负责文档格式验证和文本内容提取，
    支持 .txt 和 .md 格式文件。
    """
    
    # 支持的文件格式
    SUPPORTED_FORMATS = {'.txt', '.md'}
    
    def __init__(self):
        """初始化文档处理器"""
        logger.info("DocumentProcessor 初始化完成")
    
    def validate_file_format(self, file_path: str) -> bool:
        """
        验证文件格式是否受支持
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为支持的格式
        """
        if not file_path:
            logger.warning("文件路径为空")
            return False
        
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            is_supported = extension in self.SUPPORTED_FORMATS
            
            if is_supported:
                logger.debug(f"文件格式验证通过: {file_path} ({extension})")
            else:
                logger.warning(f"不支持的文件格式: {file_path} ({extension})")
                
            return is_supported
            
        except Exception as e:
            logger.error(f"文件格式验证失败: {file_path}, 错误: {e}")
            return False
    
    def detect_file_format(self, file_path: str) -> Optional[str]:
        """
        检测文件格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件格式（不含点号），如 'txt', 'md'，失败时返回 None
        """
        if not self.validate_file_format(file_path):
            return None
        
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            return extension[1:]  # 去掉点号
            
        except Exception as e:
            logger.error(f"文件格式检测失败: {file_path}, 错误: {e}")
            return None
    
    def extract_text(self, file_path: str) -> Tuple[bool, str]:
        """
        从文件中提取文本内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            (是否成功, 提取的文本内容)
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False, ""
        
        if not self.validate_file_format(file_path):
            logger.error(f"不支持的文件格式: {file_path}")
            return False, ""
        
        file_format = self.detect_file_format(file_path)
        
        try:
            if file_format == 'txt':
                return self._extract_txt_content(file_path)
            elif file_format == 'md':
                return self._extract_md_content(file_path)
            else:
                logger.error(f"未知的文件格式: {file_format}")
                return False, ""
                
        except Exception as e:
            logger.error(f"文本提取失败: {file_path}, 错误: {e}")
            return False, ""
    
    def _extract_txt_content(self, file_path: str) -> Tuple[bool, str]:
        """
        提取 .txt 文件内容
        
        使用 UTF-8 编码读取文本文件，
        如果失败则尝试其他常见编码。
        
        Args:
            file_path: .txt 文件路径
            
        Returns:
            (是否成功, 文本内容)
        """
        # 尝试的编码列表，优先使用 UTF-8
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    
                logger.info(f"成功读取 .txt 文件: {file_path} (编码: {encoding})")
                return True, content.strip()
                
            except UnicodeDecodeError:
                logger.debug(f"编码 {encoding} 读取失败，尝试下一个编码")
                continue
            except Exception as e:
                logger.error(f"读取 .txt 文件失败: {file_path}, 编码: {encoding}, 错误: {e}")
                continue
        
        logger.error(f"所有编码都无法读取文件: {file_path}")
        return False, ""
    
    def _extract_md_content(self, file_path: str) -> Tuple[bool, str]:
        """
        提取 .md 文件的纯文本内容
        
        移除 Markdown 标记，提取纯文本内容。
        
        Args:
            file_path: .md 文件路径
            
        Returns:
            (是否成功, 纯文本内容)
        """
        # 首先读取原始内容
        success, raw_content = self._extract_txt_content(file_path)
        
        if not success:
            return False, ""
        
        try:
            # 移除 Markdown 标记，提取纯文本
            plain_text = self._remove_markdown_syntax(raw_content)
            
            logger.info(f"成功提取 .md 文件纯文本: {file_path}")
            return True, plain_text.strip()
            
        except Exception as e:
            logger.error(f"Markdown 文本提取失败: {file_path}, 错误: {e}")
            return False, ""
    
    def _remove_markdown_syntax(self, markdown_text: str) -> str:
        """
        移除 Markdown 语法标记
        
        Args:
            markdown_text: 原始 Markdown 文本
            
        Returns:
            移除标记后的纯文本
        """
        text = markdown_text
        
        # 移除代码块 (```)
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # 移除行内代码 (`)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # 移除标题标记 (#)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # 移除粗体和斜体标记 (**, *, __)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **粗体**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *斜体*
        text = re.sub(r'__([^_]+)__', r'\1', text)      # __粗体__
        text = re.sub(r'_([^_]+)_', r'\1', text)        # _斜体_
        
        # 移除删除线 (~~)
        text = re.sub(r'~~([^~]+)~~', r'\1', text)
        
        # 移除链接 [文本](URL)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # 移除图片 ![alt](URL)
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        
        # 移除引用标记 (>)
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        
        # 移除列表标记 (-, *, +, 数字.)
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # 移除水平分割线 (---, ***)
        text = re.sub(r'^[-*]{3,}$', '', text, flags=re.MULTILINE)
        
        # 移除表格分隔符
        text = re.sub(r'\|', ' ', text)
        text = re.sub(r'^[-:\s|]+$', '', text, flags=re.MULTILINE)
        
        # 清理多余的空白字符
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 多个空行合并为两个
        text = re.sub(r'[ \t]+', ' ', text)      # 多个空格合并为一个
        
        return text
    
    def get_file_info(self, file_path: str) -> dict:
        """
        获取文件基本信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含文件信息的字典
        """
        info = {
            'path': file_path,
            'exists': False,
            'format': None,
            'size': 0,
            'supported': False
        }
        
        try:
            if os.path.exists(file_path):
                info['exists'] = True
                info['size'] = os.path.getsize(file_path)
                info['format'] = self.detect_file_format(file_path)
                info['supported'] = self.validate_file_format(file_path)
                
        except Exception as e:
            logger.error(f"获取文件信息失败: {file_path}, 错误: {e}")
        
        return info