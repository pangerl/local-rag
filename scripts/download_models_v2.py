#!/usr/bin/env python3
"""
Local RAG 模型自动下载脚本
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(repo_id: str, local_dir: Path, description: str):
    """下载单个模型"""
    try:
        logger.info(f"开始下载 {description}...")
        logger.info(f"仓库: {repo_id}")
        logger.info(f"目标目录: {local_dir}")

        # 创建目录
        local_dir.mkdir(parents=True, exist_ok=True)

        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        logger.info(f"✅ {description} 下载完成")
        return True

    except Exception as e:
        logger.error(f"❌ {description} 下载失败: {e}")
        return False

def verify_model(model_dir: Path, model_name: str):
    """验证模型文件完整性"""
    required_files = [
        "config.json",
        # "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        logger.warning(f"⚠️  {model_name} 缺少文件: {missing_files}")
        return False
    else:
        logger.info(f"✅ {model_name} 文件完整")
        return True

def main():
    """主函数"""
    logger.info("开始下载 Local RAG 系统模型...")

    # 模型配置
    models = [
        {
            "repo_id": "BAAI/bge-m3",
            "local_dir": Path("models/bge-m3"),
            "description": "中文嵌入模型 (bge-m3)"
        },
        {
            "repo_id": "BAAI/bge-reranker-v2-m3",
            "local_dir": Path("models/bge-reranker-v2-m3"),
            "description": "重排序模型 (bge-reranker-v2-m3)"
        }
    ]

    # 检查网络连接
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code != 200:
            logger.error("❌ 无法连接到 Hugging Face，请检查网络连接")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 网络连接检查失败: {e}")
        sys.exit(1)

    # 下载模型
    success_count = 0
    for model in models:
        if download_model(model["repo_id"], model["local_dir"], model["description"]):
            if verify_model(model["local_dir"], model["description"]):
                success_count += 1

    # 总结
    logger.info(f"模型下载完成: {success_count}/{len(models)} 成功")

    if success_count == len(models):
        logger.info("🎉 所有模型下载成功！系统已准备就绪。")

        # 显示模型信息
        logger.info("\n📊 模型信息:")
        for model in models:
            model_dir = model["local_dir"]
            if model_dir.exists():
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                logger.info(f"  - {model['description']}: {size_mb:.1f} MB")
    else:
        logger.error("❌ 部分模型下载失败，请检查错误信息并重试")
        sys.exit(1)

if __name__ == "__main__":
    main()
