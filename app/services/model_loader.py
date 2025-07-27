"""
模型加载器模块
负责从本地路径加载嵌入模型和重排序模型，确保完全离线运行
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.core.exceptions import ModelLoadError


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    本地模型加载器
    
    负责从指定本地路径加载预训练的嵌入模型和重排序模型，
    确保不发起任何网络请求，严格本地加载
    """
    
    def __init__(self, settings: Settings):
        """
        初始化模型加载器
        
        Args:
            settings: 系统配置对象
        """
        self.settings = settings
        self.embedding_model: Optional[SentenceTransformer] = None
        self.reranker_model: Optional[SentenceTransformer] = None
        self._models_loaded = False
        
        logger.info(f"模型加载器初始化完成，嵌入模型路径: {self.settings.embedding_model_path}")
        logger.info(f"重排序模型路径: {self.settings.reranker_model_path}")
    
    def validate_model_files(self) -> Dict[str, bool]:
        """
        验证模型文件是否存在且完整
        
        Returns:
            Dict[str, bool]: 验证结果字典，包含各模型的验证状态
            
        Raises:
            ModelLoadError: 当关键模型文件缺失时抛出异常
        """
        validation_results = {}
        
        # 验证嵌入模型目录和关键文件
        embedding_path = self.settings.embedding_model_path
        validation_results['embedding_model_dir'] = embedding_path.exists() and embedding_path.is_dir()
        
        if validation_results['embedding_model_dir']:
            # 检查关键模型文件
            config_file = embedding_path / "config.json"
            model_file = embedding_path / "pytorch_model.bin"
            safetensors_file = embedding_path / "model.safetensors"
            
            validation_results['embedding_config'] = config_file.exists()
            validation_results['embedding_weights'] = model_file.exists() or safetensors_file.exists()
        else:
            validation_results['embedding_config'] = False
            validation_results['embedding_weights'] = False
        
        # 验证重排序模型目录和关键文件
        reranker_path = self.settings.reranker_model_path
        validation_results['reranker_model_dir'] = reranker_path.exists() and reranker_path.is_dir()
        
        if validation_results['reranker_model_dir']:
            # 检查关键模型文件
            config_file = reranker_path / "config.json"
            model_file = reranker_path / "pytorch_model.bin"
            safetensors_file = reranker_path / "model.safetensors"
            
            validation_results['reranker_config'] = config_file.exists()
            validation_results['reranker_weights'] = model_file.exists() or safetensors_file.exists()
        else:
            validation_results['reranker_config'] = False
            validation_results['reranker_weights'] = False
        
        # 记录验证结果
        logger.info(f"模型文件验证结果: {validation_results}")
        
        # 检查是否有关键文件缺失
        missing_files = []
        if not validation_results['embedding_model_dir']:
            missing_files.append(f"嵌入模型目录: {embedding_path}")
        elif not validation_results['embedding_config']:
            missing_files.append(f"嵌入模型配置文件: {embedding_path}/config.json")
        elif not validation_results['embedding_weights']:
            missing_files.append(f"嵌入模型权重文件: {embedding_path}/pytorch_model.bin 或 model.safetensors")
            
        if not validation_results['reranker_model_dir']:
            missing_files.append(f"重排序模型目录: {reranker_path}")
        elif not validation_results['reranker_config']:
            missing_files.append(f"重排序模型配置文件: {reranker_path}/config.json")
        elif not validation_results['reranker_weights']:
            missing_files.append(f"重排序模型权重文件: {reranker_path}/pytorch_model.bin 或 model.safetensors")
        
        if missing_files:
            error_msg = f"模型文件验证失败，缺失文件: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        
        return validation_results
    
    def load_embedding_model(self) -> SentenceTransformer:
        """
        加载嵌入模型，严格从本地路径加载
        
        Returns:
            SentenceTransformer: 加载的嵌入模型实例
            
        Raises:
            ModelLoadError: 当模型加载失败时抛出异常
        """
        if self.embedding_model is not None:
            logger.info("嵌入模型已加载，返回缓存实例")
            return self.embedding_model
        
        try:
            model_path = str(self.settings.embedding_model_path)
            logger.info(f"开始加载嵌入模型: {model_path}")
            
            # 设置环境变量禁用网络请求
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            # 从本地路径加载模型
            self.embedding_model = SentenceTransformer(
                model_path,
                device='cpu',  # 默认使用 CPU，可根据需要调整
                cache_folder=None  # 禁用缓存文件夹
            )
            
            logger.info(f"嵌入模型加载成功: {model_path}")
            return self.embedding_model
            
        except Exception as e:
            error_msg = f"嵌入模型加载失败: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def load_reranker_model(self) -> SentenceTransformer:
        """
        加载重排序模型，严格从本地路径加载
        
        Returns:
            SentenceTransformer: 加载的重排序模型实例
            
        Raises:
            ModelLoadError: 当模型加载失败时抛出异常
        """
        if self.reranker_model is not None:
            logger.info("重排序模型已加载，返回缓存实例")
            return self.reranker_model
        
        try:
            model_path = str(self.settings.reranker_model_path)
            logger.info(f"开始加载重排序模型: {model_path}")
            
            # 设置环境变量禁用网络请求
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            # 从本地路径加载模型
            self.reranker_model = SentenceTransformer(
                model_path,
                device='cpu',  # 默认使用 CPU，可根据需要调整
                cache_folder=None  # 禁用缓存文件夹
            )
            
            logger.info(f"重排序模型加载成功: {model_path}")
            return self.reranker_model
            
        except Exception as e:
            error_msg = f"重排序模型加载失败: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def load_all_models(self) -> Dict[str, SentenceTransformer]:
        """
        加载所有模型
        
        Returns:
            Dict[str, SentenceTransformer]: 包含所有加载模型的字典
            
        Raises:
            ModelLoadError: 当任何模型加载失败时抛出异常
        """
        logger.info("开始加载所有模型")
        
        # 首先验证模型文件
        self.validate_model_files()
        
        try:
            # 加载嵌入模型
            embedding_model = self.load_embedding_model()
            
            # 加载重排序模型
            reranker_model = self.load_reranker_model()
            
            self._models_loaded = True
            logger.info("所有模型加载完成")
            
            return {
                'embedding': embedding_model,
                'reranker': reranker_model
            }
            
        except Exception as e:
            error_msg = f"模型加载过程中发生错误: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def is_models_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 如果所有模型都已加载返回 True，否则返回 False
        """
        return self._models_loaded and self.embedding_model is not None and self.reranker_model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 包含模型信息的字典
        """
        info = {
            'embedding_model_path': str(self.settings.embedding_model_path),
            'reranker_model_path': str(self.settings.reranker_model_path),
            'models_loaded': self.is_models_loaded(),
            'embedding_model_loaded': self.embedding_model is not None,
            'reranker_model_loaded': self.reranker_model is not None
        }
        
        if self.embedding_model is not None:
            info['embedding_model_device'] = str(self.embedding_model.device)
            
        if self.reranker_model is not None:
            info['reranker_model_device'] = str(self.reranker_model.device)
        
        return info
    
    def unload_models(self):
        """
        卸载所有模型，释放内存
        """
        logger.info("开始卸载模型")
        
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
            logger.info("嵌入模型已卸载")
        
        if self.reranker_model is not None:
            del self.reranker_model
            self.reranker_model = None
            logger.info("重排序模型已卸载")
        
        self._models_loaded = False
        logger.info("所有模型卸载完成")