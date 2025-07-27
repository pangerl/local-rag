"""
模型加载器单元测试
使用模拟模型测试 ModelLoader 的功能
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.services.model_loader import ModelLoader
from app.core.exceptions import ModelLoadError


class TestModelLoader:
    """ModelLoader 测试类"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """创建临时模型目录用于测试"""
        temp_dir = tempfile.mkdtemp()
        
        # 创建嵌入模型目录和文件
        embedding_dir = Path(temp_dir) / "bge-small-zh-v1.5"
        embedding_dir.mkdir(parents=True)
        (embedding_dir / "config.json").write_text('{"model_type": "bert"}')
        (embedding_dir / "pytorch_model.bin").write_text("fake model weights")
        
        # 创建重排序模型目录和文件
        reranker_dir = Path(temp_dir) / "bge-reranker-base"
        reranker_dir.mkdir(parents=True)
        (reranker_dir / "config.json").write_text('{"model_type": "bert"}')
        (reranker_dir / "model.safetensors").write_text("fake model weights")
        
        yield temp_dir
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_settings(self, temp_model_dir):
        """创建测试用的配置对象"""
        settings = Settings()
        settings.MODEL_BASE_PATH = temp_model_dir
        return settings
    
    @pytest.fixture
    def model_loader(self, test_settings):
        """创建 ModelLoader 实例"""
        return ModelLoader(test_settings)
    
    def test_init(self, model_loader, test_settings):
        """测试 ModelLoader 初始化"""
        assert model_loader.settings == test_settings
        assert model_loader.embedding_model is None
        assert model_loader.reranker_model is None
        assert model_loader._models_loaded is False
    
    def test_validate_model_files_success(self, model_loader):
        """测试模型文件验证成功的情况"""
        result = model_loader.validate_model_files()
        
        assert result['embedding_model_dir'] is True
        assert result['embedding_config'] is True
        assert result['embedding_weights'] is True
        assert result['reranker_model_dir'] is True
        assert result['reranker_config'] is True
        assert result['reranker_weights'] is True
    
    def test_validate_model_files_missing_embedding_dir(self, test_settings):
        """测试嵌入模型目录缺失的情况"""
        # 使用不存在的路径
        test_settings.MODEL_BASE_PATH = "/nonexistent/path"
        model_loader = ModelLoader(test_settings)
        
        with pytest.raises(ModelLoadError) as exc_info:
            model_loader.validate_model_files()
        
        assert "嵌入模型目录" in str(exc_info.value)
    
    def test_validate_model_files_missing_config(self, temp_model_dir, test_settings):
        """测试配置文件缺失的情况"""
        # 删除嵌入模型的配置文件
        config_file = Path(temp_model_dir) / "bge-small-zh-v1.5" / "config.json"
        config_file.unlink()
        
        model_loader = ModelLoader(test_settings)
        
        with pytest.raises(ModelLoadError) as exc_info:
            model_loader.validate_model_files()
        
        assert "嵌入模型配置文件" in str(exc_info.value)
    
    def test_validate_model_files_missing_weights(self, temp_model_dir, test_settings):
        """测试权重文件缺失的情况"""
        # 删除嵌入模型的权重文件
        weight_file = Path(temp_model_dir) / "bge-small-zh-v1.5" / "pytorch_model.bin"
        weight_file.unlink()
        
        model_loader = ModelLoader(test_settings)
        
        with pytest.raises(ModelLoadError) as exc_info:
            model_loader.validate_model_files()
        
        assert "嵌入模型权重文件" in str(exc_info.value)
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_load_embedding_model_success(self, mock_sentence_transformer, model_loader):
        """测试嵌入模型加载成功"""
        # 创建模拟的 SentenceTransformer 实例
        mock_model = Mock(spec=SentenceTransformer)
        mock_sentence_transformer.return_value = mock_model
        
        result = model_loader.load_embedding_model()
        
        assert result == mock_model
        assert model_loader.embedding_model == mock_model
        
        # 验证调用参数
        mock_sentence_transformer.assert_called_once()
        call_args = mock_sentence_transformer.call_args
        assert str(model_loader.settings.embedding_model_path) in call_args[0]
        assert call_args[1]['device'] == 'cpu'
        assert call_args[1]['cache_folder'] is None
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_load_embedding_model_cached(self, mock_sentence_transformer, model_loader):
        """测试嵌入模型缓存功能"""
        # 创建模拟的 SentenceTransformer 实例
        mock_model = Mock(spec=SentenceTransformer)
        model_loader.embedding_model = mock_model
        
        result = model_loader.load_embedding_model()
        
        assert result == mock_model
        # 验证没有调用 SentenceTransformer 构造函数
        mock_sentence_transformer.assert_not_called()
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_load_embedding_model_failure(self, mock_sentence_transformer, model_loader):
        """测试嵌入模型加载失败"""
        # 模拟加载失败
        mock_sentence_transformer.side_effect = Exception("模型加载失败")
        
        with pytest.raises(ModelLoadError) as exc_info:
            model_loader.load_embedding_model()
        
        assert "嵌入模型加载失败" in str(exc_info.value)
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_load_reranker_model_success(self, mock_sentence_transformer, model_loader):
        """测试重排序模型加载成功"""
        # 创建模拟的 SentenceTransformer 实例
        mock_model = Mock(spec=SentenceTransformer)
        mock_sentence_transformer.return_value = mock_model
        
        result = model_loader.load_reranker_model()
        
        assert result == mock_model
        assert model_loader.reranker_model == mock_model
        
        # 验证调用参数
        mock_sentence_transformer.assert_called_once()
        call_args = mock_sentence_transformer.call_args
        assert str(model_loader.settings.reranker_model_path) in call_args[0]
        assert call_args[1]['device'] == 'cpu'
        assert call_args[1]['cache_folder'] is None
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_load_reranker_model_cached(self, mock_sentence_transformer, model_loader):
        """测试重排序模型缓存功能"""
        # 创建模拟的 SentenceTransformer 实例
        mock_model = Mock(spec=SentenceTransformer)
        model_loader.reranker_model = mock_model
        
        result = model_loader.load_reranker_model()
        
        assert result == mock_model
        # 验证没有调用 SentenceTransformer 构造函数
        mock_sentence_transformer.assert_not_called()
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_load_reranker_model_failure(self, mock_sentence_transformer, model_loader):
        """测试重排序模型加载失败"""
        # 模拟加载失败
        mock_sentence_transformer.side_effect = Exception("模型加载失败")
        
        with pytest.raises(ModelLoadError) as exc_info:
            model_loader.load_reranker_model()
        
        assert "重排序模型加载失败" in str(exc_info.value)
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_load_all_models_success(self, mock_sentence_transformer, model_loader):
        """测试加载所有模型成功"""
        # 创建模拟的 SentenceTransformer 实例
        mock_embedding_model = Mock(spec=SentenceTransformer)
        mock_reranker_model = Mock(spec=SentenceTransformer)
        
        # 配置 mock 返回不同的实例
        mock_sentence_transformer.side_effect = [mock_embedding_model, mock_reranker_model]
        
        result = model_loader.load_all_models()
        
        assert result['embedding'] == mock_embedding_model
        assert result['reranker'] == mock_reranker_model
        assert model_loader._models_loaded is True
        assert model_loader.is_models_loaded() is True
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_load_all_models_failure(self, mock_sentence_transformer, model_loader):
        """测试加载所有模型失败"""
        # 模拟第一个模型加载成功，第二个失败
        mock_embedding_model = Mock(spec=SentenceTransformer)
        mock_sentence_transformer.side_effect = [mock_embedding_model, Exception("重排序模型加载失败")]
        
        with pytest.raises(ModelLoadError) as exc_info:
            model_loader.load_all_models()
        
        assert "模型加载过程中发生错误" in str(exc_info.value)
        assert model_loader._models_loaded is False
    
    def test_is_models_loaded_false_initially(self, model_loader):
        """测试初始状态下模型未加载"""
        assert model_loader.is_models_loaded() is False
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_is_models_loaded_true_after_loading(self, mock_sentence_transformer, model_loader):
        """测试加载后模型状态为已加载"""
        # 创建模拟的 SentenceTransformer 实例
        mock_embedding_model = Mock(spec=SentenceTransformer)
        mock_reranker_model = Mock(spec=SentenceTransformer)
        mock_sentence_transformer.side_effect = [mock_embedding_model, mock_reranker_model]
        
        model_loader.load_all_models()
        
        assert model_loader.is_models_loaded() is True
    
    def test_get_model_info_initial(self, model_loader):
        """测试获取初始模型信息"""
        info = model_loader.get_model_info()
        
        assert 'embedding_model_path' in info
        assert 'reranker_model_path' in info
        assert info['models_loaded'] is False
        assert info['embedding_model_loaded'] is False
        assert info['reranker_model_loaded'] is False
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_get_model_info_after_loading(self, mock_sentence_transformer, model_loader):
        """测试加载后获取模型信息"""
        # 创建模拟的 SentenceTransformer 实例
        mock_embedding_model = Mock(spec=SentenceTransformer)
        mock_reranker_model = Mock(spec=SentenceTransformer)
        mock_embedding_model.device = "cpu"
        mock_reranker_model.device = "cpu"
        mock_sentence_transformer.side_effect = [mock_embedding_model, mock_reranker_model]
        
        model_loader.load_all_models()
        info = model_loader.get_model_info()
        
        assert info['models_loaded'] is True
        assert info['embedding_model_loaded'] is True
        assert info['reranker_model_loaded'] is True
        assert info['embedding_model_device'] == "cpu"
        assert info['reranker_model_device'] == "cpu"
    
    @patch('app.services.model_loader.SentenceTransformer')
    def test_unload_models(self, mock_sentence_transformer, model_loader):
        """测试卸载模型"""
        # 先加载模型
        mock_embedding_model = Mock(spec=SentenceTransformer)
        mock_reranker_model = Mock(spec=SentenceTransformer)
        mock_sentence_transformer.side_effect = [mock_embedding_model, mock_reranker_model]
        
        model_loader.load_all_models()
        assert model_loader.is_models_loaded() is True
        
        # 卸载模型
        model_loader.unload_models()
        
        assert model_loader.embedding_model is None
        assert model_loader.reranker_model is None
        assert model_loader._models_loaded is False
        assert model_loader.is_models_loaded() is False
    
    def test_environment_variables_set(self, model_loader):
        """测试环境变量设置"""
        with patch('app.services.model_loader.SentenceTransformer') as mock_st:
            mock_model = Mock(spec=SentenceTransformer)
            mock_st.return_value = mock_model
            
            model_loader.load_embedding_model()
            
            # 验证环境变量被设置
            assert os.environ.get('TRANSFORMERS_OFFLINE') == '1'
            assert os.environ.get('HF_HUB_OFFLINE') == '1'