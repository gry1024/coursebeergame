"""Global random seed control for reproducible experiments."""

import random
import numpy as np
import torch


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs.

    Also disables CuDNN's auto-tuner so that GPU convolutions are
    deterministic (slower but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False