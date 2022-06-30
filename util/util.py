import random
import torch
from util import types


def set_seed(seed: int):
    """Set seed for python and pytorch
    """
    random.seed(seed)
    torch.manual_seed(seed)
