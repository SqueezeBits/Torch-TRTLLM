from typing import Any

import tensorrt as trt
import torch

from .plugin import Plugin


class GemmPlugin(Plugin):
    """TensorRT plugin for matrix multiplication (GEMM) operations.

    Attributes:
        transa (int): Whether to transpose first input matrix. Defaults to 0.
        transb (int): Whether to transpose second input matrix. Defaults to 0.
        pad_lda (int): Padding for leading dimension of first matrix. Defaults to 0.
        pad_ldb (int): Padding for leading dimension of second matrix. Defaults to 0.
        pad_ldc (int): Padding for leading dimension of output matrix. Defaults to 0.
        type_id (trt.DataType): Data type for computation.
        use_fp8 (int): Whether to use FP8 precision. Defaults to 0.
        alpha (float): Scaling factor for multiplication. Defaults to 1.0.
    """

    # the order of the attributes does matter!
    transa: int = 0
    transb: int = 0
    pad_lda: int = 0
    pad_ldb: int = 0
    pad_ldc: int = 0
    type_id: trt.DataType
    use_fp8: int = 0
    alpha: float = 1.0

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform matrix multiplication between input tensors.

        Args:
            x (torch.Tensor): First input tensor
            weight (torch.Tensor): Second input tensor (weight matrix)
            **kwargs (Any): Additional keyword arguments

        Returns:
            torch.Tensor: Result of matrix multiplication
        """
        return x @ weight.transpose(1, 0)
