#!/bin/bash

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 显示提示信息
echo "✅ 虚拟环境已激活"
echo "🐍 Python 版本: $(python --version)"
echo "📦 PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "使用 'deactivate' 命令退出虚拟环境"
echo ""

# 启动新的shell会话保持激活状态
exec $SHELL 