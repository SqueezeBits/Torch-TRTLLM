import operator

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torch.fx.node import Argument, Target
from torch_tensorrt import dtype

from ..fx.utils import get_tensor_metadata


def build_onnx_from_fx(graph_module: GraphModule) -> onnx.ModelProto:
    inputs: dict[str, gs.Tensor] = {}
    outputs: list[gs.Tensor] = []
    tensors: dict[str, gs.Tensor] = {}
    nodes: dict[str, gs.Node] = {}
    const_args: dict[str, gs.Constant] = {}

    def create_variable(node: Node, name: str | None = None) -> gs.Variable:
        assert (meta := get_tensor_metadata(node))
        return gs.Variable(
            name or node.name,
            dtype._from(meta.dtype).to(np.dtype),
            tuple(-1 if isinstance(s, torch.SymInt) else s for s in meta.shape),
        )

    def create_constant(node: Node, name: str | None = None) -> gs.Constant:
        assert (meta := get_tensor_metadata(node))
        return gs.Constant(
            name or node.name,
            torch.zeros(meta.shape, dtype=meta.dtype).numpy(),
        )

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
        elif node.op == "get_attr":
            tensor = create_constant(node)
            tensors[node.name] = tensor
        elif node.op == "call_function" and callable(target := node.target):
            if is_multi_output_getitem(node):
                continue
            all_inputs = {f"args_{i}": arg for i, arg in enumerate(node.args)}
            all_inputs.update(node.kwargs)
            input_tensors = [
                tensors[x.name] if isinstance(x, Node) else convert_argument_as_constant(x) for x in all_inputs.values()
            ]
            getitems = [user for user in node.users if user.op == "call_function" and user.target is operator.getitem]
            output_tensors: list[gs.Tensor] = []
            if len(getitems) == len(node.users) and len(getitems) > 1:
                for i, getitem in enumerate(getitems):
                    output_tensor = create_variable(getitem, name=f"{node.name}_output_{i}")
                    output_tensors.append(output_tensor)
                    tensors[getitem.name] = output_tensor
            else:
                output_tensor = create_variable(node)
                output_tensors.append(output_tensor)
                tensors[node.name] = output_tensor

            nodes[node.name] = gs.Node(
                op=get_target_name(target),
                name=node.name,
                inputs=input_tensors,
                outputs=output_tensors,
                attrs={k: v for k, v in node.meta.items() if isinstance(v, int | float | bool | str | dict)},
            )

    graph = gs.Graph(
        nodes=list(nodes.values()),
        inputs=list(inputs.values()),
        outputs=outputs,
        name=graph_module.meta.get("name"),
    )
    graph.cleanup()
    graph.toposort()
    return gs.export_onnx(graph, do_type_check=False)


def is_multi_output_getitem(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is operator.getitem
        and len(node.all_input_nodes) == 1
        and len(siblings := node.all_input_nodes[0].users) > 1
        and all(sibling.op == "call_function" and sibling.target is operator.getitem for sibling in siblings)
    )


def get_target_name(f: Target) -> str:
    if isinstance(f, OpOverload):
        return str(f).replace(".", "::")
    if callable(f):
        return f.__name__
    return str(f)
