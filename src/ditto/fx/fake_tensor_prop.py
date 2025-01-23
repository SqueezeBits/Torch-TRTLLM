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
from typing import Any

import torch
import torch.utils._pytree as pytree
from loguru import logger
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node

from .nodes import GetAttr
from .utils import get_fake_mode


@contextlib.contextmanager
def fake_tensor_prop_on_node_creation(graph_module: GraphModule) -> Generator[None, None, None]:
    graph_module._register_create_node_hook(run_fake_tensor_prop)
    try:
        yield None
    finally:
        graph_module._unregister_create_node_hook(run_fake_tensor_prop)


def run_fake_tensor_prop(node: Node) -> None:
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
    with fake_mode:
        logger.trace(f"Running fake tensor prop for {node}")
        node.output = node.tensor.clone()  # type: ignore[assignment]
        logger.trace(f"Fake tensor prop result: {node.output}")


def run_fake_call_function_prop(fake_mode: FakeTensorMode, node: Node) -> None:
    assert node.op == "call_function" and callable(node.target)
    flat_args, spec = pytree.tree_flatten((node.args, node.kwargs))
    flat_values: list[Any] = []
    for x in flat_args:
        if isinstance(x, Node):
            val = x.meta.get("val")
            if not isinstance(val, torch.SymInt | FakeTensor):
                logger.error(
                    f"Failed to find fake tensor or symbolic integer value from input {x.format_node()} "
                    f"while running shape inference for {node.format_node()} ({val=})"
                )
                return
            if isinstance(val, FakeTensor):
                if val.fake_mode is not fake_mode:
                    logger.error(f"{val=} from {x.format_node} belongs is defined under different fake mode.")
                    return
                with fake_mode:
                    val = val.cpu()
            flat_values.append(val)
        else:
            flat_values.append(x)

    args_, kwargs_ = pytree.tree_unflatten(flat_values, spec)
    with fake_mode:
        logger.trace(f"Running fake tensor prop for {node.format_node()} with {args_=}, {kwargs_=}")
        output = node.target(*args_, **kwargs_)
        logger.trace(f"Fake tensor prop result: {output}")
    node.meta["val"] = output
