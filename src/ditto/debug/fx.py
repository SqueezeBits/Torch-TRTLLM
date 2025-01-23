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

import json
import operator
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import TensorProto
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torch.fx.node import Argument, Target

from ..constants import DEBUG_TENSOR_CHUNK_SIZE
from ..fx.utils import get_tensor_metadata
from ..types import DataType
from .constant import make_constant


def build_onnx_from_fx(graph_module: GraphModule) -> onnx.ModelProto:
    inputs: dict[str, gs.Tensor] = {}
    outputs: list[gs.Tensor] = []
    tensors: dict[str, gs.Tensor] = {}
    nodes: dict[str, gs.Node] = {}
    const_args: dict[str, gs.Constant] = {}

    def find_tensor(node: Node) -> gs.Tensor:
        if node.op == "get_attr" and isinstance(target := node.target, str):
            return tensors[target]
        return tensors[node.name]

    def create_variable(node: Node, name: str | None = None) -> gs.Variable:
        shape: tuple[int | str, ...] | None = None
        dtype: int | None = None
        if meta := get_tensor_metadata(node):
            shape = tuple(str(s) if isinstance(s, torch.SymInt) else s for s in meta.shape)
            dtype = DataType(meta.dtype).to(TensorProto.DataType)
        return gs.Variable(name or node.name, dtype, shape)

    def create_constant(
        node: Node,
        target: str,
        *,
        param: torch.nn.Parameter | torch.Tensor | None = None,
    ) -> gs.Tensor:
        if target in tensors:
            return tensors[target]

        if param is not None:
            return make_constant(target, param)

        if (meta := get_tensor_metadata(node)) is not None:
            return make_constant(target, shape=meta.shape, dtype=meta.dtype)

        return gs.Variable(target)

    def convert_argument_as_constant(a: Argument) -> gs.Constant:
        if isinstance(a, int | float | bool):
            if isinstance(a, int) and not isinstance(a, bool):
                name = f"i:{a}"
            elif isinstance(a, float):
                name = f"f:{a:.6e}"
            else:
                name = f"b:{'true' if a else 'false'}"
            if name not in const_args:
                const_args[name] = gs.Constant(name, np.array(a))
        elif isinstance(a, str):
            name = f"s:{a}"
            if name not in const_args:
                const_args[name] = gs.Constant(name, np.array([], dtype=bool))
        elif a is None:
            name = "None"
            if name not in const_args:
                const_args[name] = gs.Constant(name, np.array([], dtype=bool))
        elif isinstance(a, list | tuple) and all(isinstance(x, int | float | bool) for x in a):
            name = f"a:{a}"
            if name not in const_args:
                const_args[name] = gs.Constant(name, np.array(a))
        else:
            name = f"{type(a).__name__}:{a}"
            if name not in const_args:
                const_args[name] = gs.Constant(name, np.array([id(a)], dtype=np.int64))
        return const_args[name]

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            tensor = create_variable(node)
            tensors[node.name] = tensor
            inputs[node.name] = tensor
        elif node.op == "output":
            outputs = [tensors[x.name] for x in node.all_input_nodes]
        elif node.op == "get_attr" and isinstance(target := node.target, str):
            try:
                param: torch.nn.Parameter | torch.Tensor = graph_module.get_parameter(target)
            except AttributeError:
                param = graph_module.get_buffer(target)
            tensors[target] = create_constant(node, target, param=param)
            if param is not None:
                param = param.flatten()
                for user in node.users:
                    if (numel := param.numel()) > 3 * DEBUG_TENSOR_CHUNK_SIZE:
                        user.meta.update(
                            {
                                f"{node.name}_first": param[:DEBUG_TENSOR_CHUNK_SIZE].tolist(),
                                f"{node.name}_middle": param[
                                    (numel - DEBUG_TENSOR_CHUNK_SIZE) // 2:(numel + DEBUG_TENSOR_CHUNK_SIZE) // 2
                                ].tolist(),
                                f"{node.name}_last": param[-DEBUG_TENSOR_CHUNK_SIZE:].tolist(),
                            }
                        )
                    else:
                        user.meta.update({node.name: param.tolist()})
        elif node.op in ("call_function", "call_module", "call_method"):
            if is_multi_output_getitem(node):
                continue
            all_inputs = {}
            for i, arg in enumerate(node.args):
                if isinstance(arg, Iterable):
                    for j, nested_arg in enumerate(arg):
                        all_inputs[f"args_{i}_{j}"] = nested_arg
                else:
                    all_inputs[f"args_{i}"] = arg
            all_inputs.update(node.kwargs)
            input_tensors = [
                find_tensor(x) if isinstance(x, Node) else convert_argument_as_constant(x) for x in all_inputs.values()
            ]
            getitems = [user for user in node.users if user.op == "call_function" and user.target is operator.getitem]
            output_tensors: list[gs.Tensor] = []
            if len(getitems) == len(node.users) and len(getitems) > 0:
                for i, getitem in enumerate(getitems):
                    output_tensor = create_variable(getitem, name=f"{node.name}_output_{i}")
                    output_tensors.append(output_tensor)
                    tensors[getitem.name] = output_tensor
            else:
                output_tensor = create_variable(node)
                output_tensors.append(output_tensor)
                tensors[node.name] = output_tensor

            nodes[node.name] = gs.Node(
                op=get_target_name(node.op, node.target, node.meta),
                name=node.name,
                inputs=input_tensors,
                outputs=output_tensors,
                attrs=process_metadata(node.meta),
            )
        else:
            raise NotImplementedError(f"The following node could not be converted: {node.format_node()}")

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, indent="··", sort_keys=True)

    graph = gs.Graph(
        nodes=list(nodes.values()),
        inputs=list(inputs.values()),
        outputs=outputs,
        name=graph_module.meta.get("name"),
    )
    graph.cleanup()
    graph.toposort()
    model_proto = gs.export_onnx(graph, do_type_check=False)
    model_proto.doc_string = json_dumps(process_metadata(graph_module.meta))
    return model_proto


def is_multi_output_getitem(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is operator.getitem
        and len(node.all_input_nodes) == 1
        and len(siblings := node.all_input_nodes[0].users) > 0
        and all(sibling.op == "call_function" and sibling.target is operator.getitem for sibling in siblings)
    )


def get_target_name(
    op: Literal["call_function", "call_module", "call_method"],
    target: Target,
    meta: dict[str, Any],
) -> str:
    match op:
        case "call_function":
            assert callable(target)
            if isinstance(target, OpOverload):
                return str(target).replace(".", "::")
            module = target.__module__
            if module.startswith("torch"):
                if module in ("torch.nn.functional", "torch._C._nn"):
                    module = "F"
                return f"{module}.{target.__name__}".replace(".", "::")
            if module.startswith("_operator"):
                return f"operator::{target.__name__}"
            return target.__name__
        case "call_method":
            assert isinstance(target, str)
            return f"Tensor::{target}"
        case "call_module":
            assert isinstance(target, str)
            if (
                isinstance(module_stack := meta.get("nn_module_stack"), dict)
                and isinstance(name_and_class := module_stack.get(target), tuple)
                and len(name_and_class) == 2
                and isinstance(original_layer_name := name_and_class[0], str)
                and issubclass(module_class := name_and_class[1], torch.nn.Module)
            ):
                original_layer_name = original_layer_name.removeprefix("L['self'].").replace(".", "/")
                return f"{original_layer_name}:{module_class.__name__}"
            return target


AllowedMetaDataType = int | float | bool | str | list[str] | list[float] | list[int] | list[bool] | TensorProto


def process_metadata(meta: dict[str, Any]) -> dict[str, AllowedMetaDataType]:
    def _is_homogeneous(seq: list | tuple) -> bool:
        return len(seq) < 2 or all(type(x) is type(seq[0]) for x in seq[1:])

    def _impl(v: Any) -> AllowedMetaDataType:
        if isinstance(v, torch.Tensor):
            shape = (*v.shape,)
            dtype = f"{v.dtype}".removeprefix("torch.")
            return f"{type(v).__name__}(shape={shape}, dtype={dtype})"
        if isinstance(v, int | float | bool | str):
            return v
        if isinstance(v, dict):
            return [f"{key}: {val}" for key, val in v.items()]
        if isinstance(v, list | tuple):
            if _is_homogeneous(v) and v and isinstance(v[0], int | float | bool | str):
                return list(v)
            return [str(x) for x in v]
        if isinstance(v, TensorProto):
            return v
        return str(v)

    return {name: _impl(value) for name, value in meta.items()}
