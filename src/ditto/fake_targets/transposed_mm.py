import torch


def fake_transposed_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mm(x, y.permute(1, 0))
