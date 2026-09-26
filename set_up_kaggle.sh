#!/bin/bash
# Kaggle 环境专属引导（只在 Kaggle 会话开头跑，本地/4090 不要执行）

pip install "kaggle-vllm[hub]==0.2.0"
kaggle-vllm bootstrap --strict
