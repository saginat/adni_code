import torch
import numpy as np
import random
import os
import math


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def lr_lambda_scheduler(current_step: int, warmup_steps: int, total_steps: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0,
        0.5
        * (
            1.0
            + math.cos(
                math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)
            )
        ),
    )
