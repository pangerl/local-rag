#!/usr/bin/env python3
"""
Local RAG 系统启动脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from app.main import run_server

if __name__ == "__main__":
    print("启动 Local RAG 系统...")
    print("按 Ctrl+C 停止服务器")
    print("-" * 50)

    try:
        run_server()
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)