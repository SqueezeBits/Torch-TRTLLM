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

import contextlib
from collections.abc import Generator

import torch
import torch.utils._pytree as pytree
from loguru import logger
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.node import Argument

from .nodes import GetAttr
from .utils import get_fake_mode


@contextlib.contextmanager
def fake_tensor_prop_on_node_creation(graph_module: GraphModule) -> Generator[None, None, None]:
    """Register fake tensor propagation hook on node creation.

    Args:
        graph_module (GraphModule): Graph module to register hook on

    Yields:
        Generator[None, None, None]: Context manager that registers and unregisters the hook
    """
    graph_module._register_create_node_hook(run_fake_tensor_prop)
    try:
        yield None
    finally:
        graph_module._unregister_create_node_hook(run_fake_tensor_prop)


def run_fake_tensor_prop(node: Node) -> None:
    """Run fake tensor propagation on a node.

    Args:
        node (Node): Node to run propagation on
    """
    if (fake_mode := get_fake_mode(node.graph)) is None:
        logger.warning(
            f"Failed to run shape inference for {node.format_node()} as as no fake mode has been found in the graph"
        )
        return

    if node.op == "call_function" and callable(node.target):
        run_fake_call_function_prop(fake_mode, node)
    elif get_attr := GetAttr.specialize_from(node):
        run_fake_get_attr_prop(fake_mode, get_attr)
    elif node.op in ("call_function", "get_attr"):
        logger.error(f"Failed to run fake tensor prop for {node.format_node()}")
    else:
        logger.trace(f"Skipping fake tensor prop for {node.format_node()}")


def run_fake_get_attr_prop(fake_mode: FakeTensorMode, node: GetAttr) -> None:
    """Run fake tensor propagation for a GetAttr node.

    Args:
        fake_mode (FakeTensorMode): Fake tensor mode to use
        node (GetAttr): GetAttr node to run propagation on
    """
    with fake_mode:
        logger.trace(f"Running fake tensor prop for {node}")
        node.output = node.tensor.clone()
        logger.trace(f"Fake tensor prop result: {node.output}")


def run_fake_call_function_prop(fake_mode: FakeTensorMode, node: Node) -> None:
    """Run fake tensor propagation for a call_function node.

    Args:
        fake_mode (FakeTensorMode): Fake tensor mode to use
        node (Node): Node to run propagation on
    """
    assert node.op == "call_function" and callable(node.target)
    with fake_mode:
        args_, kwargs_ = pytree.tree_map(as_value_on_cpu, (node.args, node.kwargs))
        logger.trace(f"Running fake tensor prop for {node.format_node()} with {args_=}, {kwargs_=}")
        output = node.target(*args_, **kwargs_)
        logger.trace(f"Fake tensor prop result: {output}")
    node.meta["val"] = output


def as_value_on_cpu(argument: Argument) -> Argument | torch.Tensor:
    """Move node argument value to CPU if it is a tensor.

    Args:
        argument (Argument): Node argument to convert

    Returns:
        Argument | torch.Tensor: CPU tensor value or original argument

    Raises:
        KeyError: If the argument is a node and has no 'val' in meta
    """
    if isinstance(argument, Node):
        if "val" not in argument.meta:
            raise KeyError(
                f"No 'val' found in the meta of the input node {argument.format_node()}, where:\n"
                f"{argument}.meta = {argument.meta}"
            )
        val = argument.meta["val"]
        return pytree.tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, val)

    return argument
