"""
模型加载器模块
负责从本地路径加载嵌入模型和重排序模型，确保完全离线运行
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

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
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        self.reranker_model: Optional[CrossEncoder] = None
        self._models_loaded = False

        logger.info(f"模型加载器初始化完成，嵌入模型路径: {self.settings.embedding_model_path}")
        logger.info(f"重排序模型路径: {self.settings.reranker_model_path}")

    def _validate_single_model_path(
        self, model_path: Path, model_type: str
    ) -> tuple[Dict[str, bool], list[str]]:
        """
        辅助函数，验证单个模型的路径和关键文件

        Args:
            model_path (Path): 模型目录的路径
            model_type (str): 模型类型 (例如, 'embedding' 或 'reranker')

        Returns:
            tuple[Dict[str, bool], list[str]]: 包含验证结果和缺失文件列表的元组
        """
        results = {}
        missing_files = []
        model_name_map = {"embedding": "嵌入", "reranker": "重排序"}
        model_name = model_name_map.get(model_type, model_type)

        dir_key = f'{model_type}_model_dir'
        config_key = f'{model_type}_config'
        weights_key = f'{model_type}_weights'

        results[dir_key] = model_path.exists() and model_path.is_dir()

        if results[dir_key]:
            config_file = model_path / "config.json"
            model_file = model_path / "pytorch_model.bin"
            safetensors_file = model_path / "model.safetensors"

            # 检查配置文件
            results[config_key] = config_file.exists()
            if not results[config_key]:
                missing_files.append(f"{model_name}模型配置文件缺失: {config_file}")

            # 检查权重文件
            results[weights_key] = model_file.exists() or safetensors_file.exists()
            if not results[weights_key]:
                missing_files.append(
                    f"{model_name}模型权重文件缺失: {model_path}/pytorch_model.bin 或 model.safetensors"
                )
        else:
            results[config_key] = False
            results[weights_key] = False
            missing_files.append(f"{model_name}模型目录不存在: {model_path}")

        return results, missing_files

    def validate_model_files(self) -> Dict[str, bool]:
        """
        验证模型文件是否存在且完整

        Returns:
            Dict[str, bool]: 验证结果字典，包含各模型的验证状态

        Raises:
            ModelLoadError: 当关键模型文件缺失时抛出异常
        """
        embedding_results, embedding_missing = self._validate_single_model_path(
            self.settings.embedding_model_path, "embedding"
        )
        reranker_results, reranker_missing = self._validate_single_model_path(
            self.settings.reranker_model_path, "reranker"
        )

        all_results = {**embedding_results, **reranker_results}
        all_missing = embedding_missing + reranker_missing

        logger.info(f"模型文件验证结果: {all_results}")

        if all_missing:
            error_msg = f"模型文件验证失败，缺失以下文件或目录: {'; '.join(all_missing)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)

        return all_results

    def load_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        加载嵌入模型，并将其包装为 LangChain 兼容的格式

        Returns:
            HuggingFaceEmbeddings: 加载并包装后的嵌入模型实例

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

            # 使用 HuggingFaceEmbeddings 包装器加载模型
            # 这会返回一个与 LangChain 兼容的 embedding 对象
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu'}, # 明确指定设备
                encode_kwargs={'normalize_embeddings': True} # 根据模型推荐进行归一化
            )

            logger.info(f"嵌入模型加载并包装成功: {model_path}")
            return self.embedding_model

        except Exception as e:
            error_msg = f"嵌入模型加载失败: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def load_reranker_model(self) -> CrossEncoder:
        """
        加载重排序模型，严格从本地路径加载

        Returns:
            CrossEncoder: 加载的重排序模型实例

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

            # 忽略特定的 FutureWarning，因为它源于依赖库内部，我们无法直接修复
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="`encoder_attention_mask` is deprecated and will be removed in version 4.55.0.*",
                    category=FutureWarning
                )
                # 使用 CrossEncoder 加载重排序模型
                self.reranker_model = CrossEncoder(
                    model_path,
                    device='cpu',  # 默认使用 CPU
                    max_length=self.settings.RERANKER_MAX_LENGTH
                )

            logger.info(f"重排序模型 (CrossEncoder) 加载成功: {model_path}")
            return self.reranker_model

        except Exception as e:
            error_msg = f"重排序模型加载失败: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def load_all_models(self) -> Dict[str, Union[HuggingFaceEmbeddings, CrossEncoder]]:
        """
        加载所有模型

        Returns:
            Dict[str, Union[HuggingFaceEmbeddings, CrossEncoder]]: 包含所有加载模型的字典

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
            info['embedding_model_device'] = str(self.embedding_model.client.device)

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
