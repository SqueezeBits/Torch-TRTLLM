# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32
from tensorrt_llm.quantization.functional import unpack_int32_into_int8
from torch.fx import Node
from transformers.utils.quantization_config import QuantizationMethod
from typing_extensions import Self

from ...quantization import GlobalQuantConfig, QuantizeMode, QuantScheme
from ...types import StrictlyTyped
from ..nodes import MM, GetAttr, MulTensorTensor, Permute
from ..targets import Dequantizer
from ..utils import find_nearest, get_val
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class WrapWeightDequantSubgraphs(NodewiseOptimizationPass):
    """Match and replace fake dequantization by a single fake_dequantize node.

    This pass supports only fake dequantization currently.

    Attributes:
        global_quant_config (GlobalQuantConfig | None): Global quantization config
        dtype (torch.dtype): The data type of the model
    """

    global_quant_config: GlobalQuantConfig | None
    dtype: torch.dtype

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            self.global_quant_config is not None
            and len(self.global_quant_config.quant_configs) == 1
            and (weight_quant_scheme := self.global_quant_config.quant_configs[0].weight_quant_scheme) is not None
            and (mm := MM.specialize_from(node))
            and self.global_quant_config.quant_configs[0].target == mm.target
            and (
                dequantize_path := TrailingDequantizePath.configure_from(
                    mm.other, weight_quant_scheme, self.global_quant_config.hf_quant_method
                )
            )
        ):
            return {}

        if weight_quant_scheme.mode == QuantizeMode.UNKNOWN:
            weight_quant_scheme.mode = dequantize_path.quantize_mode
        weight_quant_scheme.has_zero_point = weight_quant_scheme.has_zero_point or dequantize_path.zero is not None

        unpacked_qweight = unpack_qweight(
            dequantize_path.qweight.tensor, dequantize_path.bits, self.global_quant_config.hf_quant_method
        )
        unpacked_zeros = (
            unpack_zeros(dequantize_path.zero.tensor, dequantize_path.bits, self.global_quant_config.hf_quant_method)
            if dequantize_path.zero is not None
            else None
        )

        with node.graph.inserting_before(dequantize_path.qweight.node):
            unpacked_qweight_getattr = GetAttr.create(
                node.graph,
                f"{dequantize_path.qweight.name}_unpacked",
                unpacked_qweight.T.contiguous() if dequantize_path.has_permuted_weight else unpacked_qweight,
            )
            unpacked_zeros_getattr = (
                GetAttr.create(node.graph, f"{dequantize_path.zero.name}_unpacked", unpacked_zeros)
                if dequantize_path.zero is not None and unpacked_zeros is not None
                else None
            )

        dequantize = Dequantizer(
            dtype=self.dtype,
            global_quant_config=self.global_quant_config,
            output_shape=dequantize_path.org_weight_shape,
            bits=dequantize_path.bits,
            mode=dequantize_path.quantize_mode,
            group_size=dequantize_path.group_size,
        )
        with node.graph.inserting_before(mm.other):
            dequantize_node = node.graph.call_function(
                dequantize,
                args=(
                    unpacked_qweight_getattr.node,
                    dequantize_path.scale.node,
                    unpacked_zeros_getattr.node if unpacked_zeros_getattr else None,
                ),
            )
        return {mm.other: ReplaceAllUses(by=dequantize_node)}


class TrailingDequantizePath(StrictlyTyped):
    """Find the root get_attr nodes of a node.

    Attributes:
        bits (int): The number of bits used to quantize the weight
        org_weight_shape (torch.Size): The shape of the original weight
        qweight (GetAttr): The get_attr node for the quantized weight
        scale (GetAttr): The get_attr node for the scale
        zero (GetAttr | None): The get_attr node for the zero point. Defaults to None.
        group_size (int | None): The group size for the quantized weight. Defaults to None.
        has_permuted_weight (bool): Whether the weight has been permuted. Defaults to False.
        quantize_mode (QuantizeMode): The quantization mode for the quantized weight.
    """

    bits: int
    org_weight_shape: torch.Size
    qweight: GetAttr
    scale: GetAttr
    zero: GetAttr | None = None
    group_size: int | None = None
    has_permuted_weight: bool = False
    quantize_mode: QuantizeMode = QuantizeMode.UNKNOWN

    @classmethod
    def configure_from(
        cls, node: Node, weight_quant_scheme: QuantScheme, quant_method: QuantizationMethod
    ) -> Self | None:
        """Configure the trailing dequantize path from a node.

        Args:
            node (Node): The node to configure the trailing dequantize path from
            weight_quant_scheme (QuantScheme): The weight quantization scheme
            quant_method (QuantizationMethod): The quantization method used to quantize the weight

        Returns:
            Self | None: The trailing dequantize path or None if no such path is found
        """
        if not (
            isinstance(org_weight_tensor := get_val(node), torch.Tensor)
            and (ancestors_with_maxdepth := get_ancestors_with_maxdepth_to_root(node))
            and (mul := find_nearest(MulTensorTensor, node))
            and (
                scale := find_nearest(
                    GetAttr,
                    mul.this if ancestors_with_maxdepth[mul.this] < ancestors_with_maxdepth[mul.other] else mul.other,
                )
            )
        ):
            return None

        group_size = get_group_size(scale.tensor.shape, org_weight_tensor.shape, quant_method)
        quantize_mode = (
            get_quantize_mode(scale.tensor.shape, org_weight_tensor.shape, quant_method)
            if group_size is None
            else QuantizeMode.PER_GROUP
        )
        qweight, zero = find_qweight_and_zero_node(
            [n for n in ancestors_with_maxdepth if n is not scale.node and GetAttr.specialize_from(n)],
            org_weight_tensor.shape,
            weight_quant_scheme,
            quant_method,
            group_size=group_size,
        )

        if (
            scale.tensor.ndim == 2
            and group_size is not None
            and scale.tensor.shape[1] == org_weight_tensor.shape[0] // group_size
        ):
            with scale.node.graph.inserting_before(scale.node):
                scale = GetAttr.create(scale.node.graph, f"{scale.name}_permuted", scale.tensor.T)

        return cls(
            bits=int(qweight.tensor.dtype.itemsize * 8 / (org_weight_tensor.numel() / qweight.tensor.numel())),
            org_weight_shape=org_weight_tensor.shape,
            qweight=qweight,
            scale=scale,
            zero=zero,
            group_size=group_size,
            has_permuted_weight=(
                (permute := Permute.specialize_from(node)) is not None and permute.ndim == 2 and permute.dims == [1, 0]
            ),
            quantize_mode=quantize_mode,
        )


def get_ancestors_with_maxdepth_to_root(node: Node) -> dict[Node, int]:
    """Get the ancestors of a node with the maximum depth to the root.

    Args:
        node (Node): The node to get the ancestors of

    Returns:
        dict[Node, int]: The ancestors of the node with the maximum depth to the root
    """

    def maxdepth_to_root(node: Node, memo: dict[Node, int]) -> int:
        """Get the maximum depth to the root of a node.

        Args:
            node (Node): The node to get the maximum depth to the root of
            memo (dict[Node, int]): The memoization dictionary

        Returns:
            int: The maximum depth to the root of the node
        """
        if node in memo:
            return memo[node]
        if len(node.all_input_nodes) == 0:
            memo[node] = 0
            return 0
        max_depth = (
            max(maxdepth_to_root(p, memo) for p in node.all_input_nodes if p.target is not torch.ops.aten.sym_size.int)
            + 1
        )
        memo[node] = max_depth
        return max_depth

    memo: dict[Node, int] = {}
    _ = maxdepth_to_root(node, memo)

    return {node: memo[node] for node in sorted(memo, reverse=True)}


def get_group_size(
    scale_shape: torch.Size, org_weight_shape: torch.Size, quant_method: QuantizationMethod
) -> int | None:
    """Get the group size for the quantized weight.

    Args:
        scale_shape (torch.Size): The shape of the scale tensor
        org_weight_shape (torch.Size): The shape of the original weight
        quant_method (QuantizationMethod): The quantization method used to quantize the weight

    Returns:
        int | None: The group size for the quantized weight or None if no such group size is found
    """
    ndim = len(scale_shape)
    if ndim not in (0, 1, 2):
        raise ValueError(f"Expected 0, 1, or 2 dimensions, got {ndim} for scale tensor when getting group size.")

    if ndim == 2:
        dim_size_for_group = scale_shape[1] if quant_method == QuantizationMethod.COMPRESSED_TENSORS else scale_shape[0]
        if dim_size_for_group != 1:
            group_size = int(org_weight_shape[0] / dim_size_for_group)
            return group_size

    return None


def get_quantize_mode(
    scale_shape: torch.Size, org_weight_shape: torch.Size, quant_method: QuantizationMethod
) -> QuantizeMode:
    """Get the quantize mode for the quantized weight.

    Args:
        scale_shape (torch.Size): The shape of the scale tensor
        org_weight_shape (torch.Size): The shape of the original weight
        quant_method (QuantizationMethod): The quantization method used to quantize the weight
    """
    ndim = len(scale_shape)
    if ndim in (0, 1):
        return QuantizeMode.PER_TENSOR
    if ndim == 2 and scale_shape[1 if quant_method == QuantizationMethod.COMPRESSED_TENSORS else 0] == 1:
        return QuantizeMode.PER_CHANNEL

    raise RuntimeError(f"Could not infer a quantization mode for {scale_shape=} and {org_weight_shape=}.")


# pylint: disable-next=too-many-branches
def find_qweight_and_zero_node(
    nodes: list[Node],
    org_weight_shape: torch.Size,
    weight_quant_scheme: QuantScheme,
    quant_method: QuantizationMethod,
    *,
    group_size: int | None = None,
) -> tuple[GetAttr, GetAttr | None]:
    """Find the qweight and zero node for the quantized weight.

    Args:
        nodes (list[Node]): The nodes to find the qweight and zero node from
        org_weight_shape (torch.Size): The shape of the original weight
        weight_quant_scheme (QuantScheme): The weight quantization scheme
        quant_method (QuantizationMethod): The quantization method used to quantize the weight
        group_size (int | None, optional): The group size for the quantized weight. Defaults to None.

    Returns:
        tuple[GetAttr, GetAttr | None]: The qweight and zero node for the quantized weight
    """
    if quant_method in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ):
        assert group_size is not None, "Group size is required for GPTQ and AWQ."
        expected_equal_dim = 1 if quant_method == QuantizationMethod.GPTQ else 0
        for node in nodes:
            assert isinstance(node_val := get_val(node), torch.Tensor), "not found tensor value from the node"
            shape = node_val.shape
            if shape[expected_equal_dim] == org_weight_shape[expected_equal_dim] and (
                qweight := GetAttr.specialize_from(node)
            ):
                break
        else:
            raise ValueError("QWeight node not found")
        nodes.remove(qweight.node)

        for node in nodes:
            assert isinstance(node_val := get_val(node), torch.Tensor), "not found tensor value from the node"
            shape = node_val.shape
            packed_ratio = org_weight_shape[1 - expected_equal_dim] / qweight.tensor.shape[1 - expected_equal_dim]
            if (
                shape[0] * group_size == org_weight_shape[0]
                and shape[1] * packed_ratio == org_weight_shape[1]
                and (zero := GetAttr.specialize_from(node))
            ):
                break
        else:
            raise ValueError("Zero node not found")

        return qweight, zero

    if quant_method == QuantizationMethod.COMPRESSED_TENSORS:
        if weight_quant_scheme.bits == 4:  # Note: weight is expected to be packed and it might have zero point
            for node in nodes:
                assert isinstance(node_val := get_val(node), torch.Tensor), "not found tensor value from the node"
                shape = node_val.shape
                if shape[0] == org_weight_shape[1] and (qweight := GetAttr.specialize_from(node)):
                    break
            else:
                raise ValueError("QWeight node not found")
            nodes.remove(qweight.node)

            zero: GetAttr | None = None  # type: ignore [no-redef]
            for node in nodes:
                assert isinstance(node_val := get_val(node), torch.Tensor), "not found tensor value from the node"
                shape = node_val.shape
                packed_ratio = org_weight_shape[0] / qweight.tensor.shape[1]
                if (
                    group_size is not None
                    and shape[0] * packed_ratio == org_weight_shape[1]
                    and shape[1] * group_size == org_weight_shape[0]
                    and (zero := GetAttr.specialize_from(node))
                ):
                    break
        else:
            for node in nodes:
                assert isinstance(node_val := get_val(node), torch.Tensor), "not found tensor value from the node"
                shape = node_val.shape
                if (
                    len(shape) == 2
                    and shape[1] == org_weight_shape[0]
                    and shape[0] != 1
                    and (qweight := GetAttr.specialize_from(node))
                ):
                    break
            else:
                raise ValueError("QWeight node not found")
            zero = None

        return qweight, zero

    raise NotImplementedError(f"Quantization for {quant_method} is not implemented yet.")


def unpack_qweight(qweight: torch.Tensor, bits: int, quant_method: QuantizationMethod) -> torch.Tensor:
    """Unpack the quantized weight tensor from int32 to int8 or uint8.

    if the weight is already unpacked, it will return the original tensor.

    Args:
        qweight (torch.Tensor): The quantized weight tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizationMethod): The quantization method used

    Returns:
        torch.Tensor: The unpacked weight tensor
    """
    assert bits in (4, 8), f"Unsupported bits: {bits=}"
    if quant_method in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ):
        if bits == 4:
            qweight = unpack_int32_into_int8(
                qweight if quant_method is QuantizationMethod.AWQ else qweight.T,
                quant_method is QuantizationMethod.AWQ,
            )
            qweight = qweight.T if quant_method is QuantizationMethod.GPTQ else qweight
            qweight = qweight - 8
            qweight[qweight < 0] += 16
            qweight = qweight.view(torch.uint8).contiguous()
        else:
            qweight = (
                qweight.T.contiguous().view(torch.uint8).T.contiguous()
                if quant_method is QuantizationMethod.GPTQ
                else qweight.view(torch.uint8).contiguous()
            )
            qweight = (qweight - 128).to(torch.int8)
    elif quant_method == QuantizationMethod.COMPRESSED_TENSORS:
        if qweight.dtype == torch.int32:
            original_shape = torch.Size([qweight.shape[0], qweight.shape[1] * (32 // bits)])
            qweight = unpack_from_int32(qweight, bits, original_shape)

            if bits == 4:
                qweight[qweight < 0] += 16
                qweight = qweight.view(torch.uint8).contiguous()
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    assert qweight.dtype.itemsize == 1, f"Wrong dtype of qweight: {qweight.dtype=}"
    return qweight


def unpack_zeros(zeros: torch.Tensor, bits: int, quant_method: QuantizationMethod) -> torch.Tensor:
    """Unpack the quantized zero point tensor.

    Args:
        zeros (torch.Tensor): The quantized zero point tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizationMethod): The quantization method used

    Returns:
        torch.Tensor: The unpacked zero point tensor
    """
    if quant_method in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ, QuantizationMethod.COMPRESSED_TENSORS):
        assert bits in (4, 8), f"Unsupported GPTQ or AWQ bits: {bits=}"
        if bits == 4:
            zeros = unpack_int32_into_int8(
                zeros.T if quant_method is QuantizationMethod.COMPRESSED_TENSORS else zeros,
                quant_method is QuantizationMethod.AWQ,
            )
        else:
            zeros = zeros.view(torch.int8)
        zeros = -zeros + 2 ** (bits - 1) - 1 * (quant_method is QuantizationMethod.GPTQ)
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    assert zeros.dtype.itemsize == 1, f"Wrong dtype of zeros: {zeros.dtype=}"
    return zeros
