import random

import numpy as np
import torch


def set_seed(seed: int):
    """固定实验中常用的随机种子。"""

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CUDA 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False