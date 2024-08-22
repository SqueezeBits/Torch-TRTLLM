import torch


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
