import operator
from collections.abc import Callable
from typing import Any

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import dtype_abbrs
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tensorrt import dtype
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from .fx.utils import get_tensor_metadata, populate_tensor_metadata


def build_fake_onnx_proto(graph_module: GraphModule) -> onnx.ModelProto:
    inputs: dict[str, gs.Tensor] = {}
    outputs: list[gs.Tensor] = []
    tensors: dict[str, gs.Tensor] = {}
    nodes: dict[str, gs.Node] = {}

    def create_variable(node: Node, name: str | None = None) -> gs.Variable:
        assert (meta := get_tensor_metadata(node))
        return gs.Variable(
            name or node.name,
            dtype._from(meta.dtype).to(np.dtype),
            (*meta.shape,),
        )

    def create_constant(node: Node, name: str | None = None) -> gs.Constant:
        assert (meta := get_tensor_metadata(node))
        return gs.Constant(
            name or node.name,
            torch.zeros(meta.shape, dtype=meta.dtype).numpy(),
        )

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
            if target is operator.getitem:
                continue
            all_inputs = {f"args_{i}": arg for i, arg in enumerate(node.args)}
            all_inputs.update(node.kwargs)
            input_tensors = [tensors[x.name] for x in all_inputs.values() if isinstance(x, Node)]
            attributes: dict[str, Any] = {
                name: value for name, value in all_inputs.items() if not isinstance(value, Node)
            }
            getitems = [user for user in node.users if user.op == "call_function" and user.target is operator.getitem]
            output_tensors: list[gs.Tensor] = []
            if len(getitems) == len(node.users) and len(getitems) > 1:
                for i, getitem in enumerate(getitems):
                    output_tensor = create_variable(node, name=f"{node.name}_output_{i}")
                    output_tensors.append(output_tensor)
                    tensors[getitem.name] = output_tensor
            else:
                output_tensor = create_variable(node)
                output_tensors.append(output_tensor)
                tensors[node.name] = output_tensor

            nodes[node.name] = gs.Node(
                op=target.__name__,
                name=node.name,
                attrs=attributes,
                inputs=input_tensors,
                outputs=output_tensors,
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


def build_fake_graph_module(network: trt.INetworkDefinition) -> GraphModule:
    graph = Graph()
    nodes: dict[str, Node] = {}

    def extract_tensor_metadata(t: trt.ITensor) -> TensorMetadata:
        return TensorMetadata(
            shape=torch.Size(t.shape[:]),
            dtype=dtype._from(t.dtype).to(torch.dtype),
            requires_grad=False,
            stride=(),
            memory_format=None,
            is_quantized=False,
            qparams={},
        )

    param_indices: dict[str, int] = {}

    def make_param_name(s: str) -> str:
        if s not in param_indices:
            param_indices[s] = len(param_indices)
        return f"constant_{param_indices[s]}"

    fake_targets: dict[str, Callable[..., Any]] = {}

    def get_fake_target(layer: trt.ILayer) -> Callable[..., Any]:
        if (layer_type := layer.type.name.lower()) in fake_targets:
            return fake_targets[layer_type]

        def fake_target(*args: Any, **kwargs: Any) -> Any:
            ...

        fake_target.__name__ = layer_type
        fake_target.__module__ = "trt"
        fake_targets[layer_type] = fake_target
        return fake_target

    def call_function(graph: Graph, target: Callable, inputs: list[trt.ITensor | None]) -> Node:
        input_nodes: list[Node | str | None] = []
        stack_trace: list[str] = []
        for i, x in enumerate(inputs):
            if x is None:
                stack_trace.append(f"{i}-th input was None")
                input_nodes.append(x)
            elif x.name not in nodes:
                stack_trace.append(f"{i}-th input {x.name} was implicitly created by torch-trt converter")
                input_nodes.append(x.name)
            else:
                input_nodes.append(nodes[x.name])
        node = graph.call_function(
            target,
            (*input_nodes,),
        )
        if stack_trace:
            add_stack_trace(node, ", ".join(stack_trace))
        return node

    graph_module = GraphModule({}, graph)

    for i in range(network.num_inputs):
        t = network.get_input(i)
        placeholder = nodes[t.name] = graph.placeholder(t.name)
        placeholder.meta["tensor_meta"] = extract_tensor_metadata(t)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        inputs = [layer.get_input(j) for j in range(layer.num_inputs)]
        outputs = [layer.get_output(j) for j in range(layer.num_outputs)]
        assert len(outputs) > 0

        fake_target = get_fake_target(layer)
        if len(outputs) == 1:
            output = outputs[0]
            tensor_meta = extract_tensor_metadata(output)
            param_name = make_param_name(layer.name)
            if layer.type == trt.LayerType.CONSTANT:
                graph_module.register_parameter(
                    param_name,
                    torch.nn.Parameter(
                        torch.zeros(size=tensor_meta.shape, dtype=tensor_meta.dtype),
                        requires_grad=False,
                    ),
                )
            output_node = nodes[outputs[0].name] = (
                graph.get_attr(param_name)
                if layer.type == trt.LayerType.CONSTANT
                else call_function(graph, fake_target, inputs)
            )
            output_node.meta["tensor_meta"] = tensor_meta
            add_stack_trace(output_node, make_fake_stack_trace_header(layer.name), prepend=True)
        else:
            outputs_node = nodes[f"{layer.name}_output"] = graph.call_function(
                fake_target,
                tuple(None if x is None else nodes.get(x.name, x.name) for x in inputs),
            )
            for i, output in enumerate(outputs):
                output_node = nodes[output.name] = graph.call_function(
                    operator.getitem,
                    (outputs_node, i),
                )
                output_node.meta["tensor_meta"] = extract_tensor_metadata(output)
                add_stack_trace(output_node, make_fake_stack_trace_header(layer.name), prepend=True)
    graph.output(tuple(nodes[network.get_output(i).name] for i in range(network.num_outputs)))
    clean_up_graph_after_modifications(graph_module)

    run_fake_constant_folding(graph_module)
    for node in graph.nodes:
        add_stack_trace(node, f"users: {len(node.users)}")
    graph_module.meta["name"] = network.name
    return graph_module


def find_constant_nodes(graph: Graph) -> list[Node]:
    """Find all constant-foldable nodes in the graph.

    Args:
        graph (Graph): the input graph

    Returns:
        list[Node]: the list containing all constant-foldable nodes in the graph.
    """
    constant_nodes: list[Node] = []
    non_constant_nodes: list[Node] = []

    def is_constant(node: Node) -> bool:
        if node in constant_nodes:
            return True
        if node in non_constant_nodes:
            return False

        if node.op == "placeholder":
            non_constant_nodes.append(node)
            return False

        is_getattr_node = node.op == "get_attr"
        has_no_input_nodes = len(node.all_input_nodes) == 0
        is_shape_node = node.op == "call_function" and callable(node.target) and node.target.__name__ == "shape"
        if any((is_getattr_node, has_no_input_nodes, is_shape_node)):
            constant_nodes.append(node)
            return True

        result: bool = all(is_constant(input_node) for input_node in node.all_input_nodes)
        if result:
            constant_nodes.append(node)
        else:
            non_constant_nodes.append(node)
        return result

    for node in graph.nodes:
        _ = is_constant(node)

    return constant_nodes


def run_fake_constant_folding(graph_module: GraphModule) -> None:
    graph = graph_module.graph
    constant_nodes = find_constant_nodes(graph)

    folded_constants: dict[Node, int] = {}

    def make_folded_constant_name(constant: Node) -> str:
        if constant not in folded_constants:
            folded_constants[node] = len(folded_constants)
        return f"folded_constant_{folded_constants[node]}"

    def make_stack_traces(node: Node) -> str:
        def _node_repr(node: Node) -> str:
            node_repr = node.name
            if tensor_meta := get_tensor_metadata(node):
                node_repr += f": {dtype_abbrs[tensor_meta.dtype]}[{', '.join(str(x) for x in tensor_meta.shape)}]"
            if node.op == "get_attr":
                return f"{node_repr} = self.{node.target}"
            target_repr = (
                f"{node.target.__module__}.{node.target.__name__}" if callable(node.target) else f"{node.target}"
            )
            args_kwargs_repr = ", ".join(
                [*(f"{arg}" for arg in node.args), *(f"{k}={v}" for k, v in node.kwargs.items())]
            )
            node_repr = f"{node_repr} = {target_repr}({args_kwargs_repr})"
            return node_repr

        def _impl(node: Node) -> list[str]:
            stack_traces: list[str] = []
            for input_node in node.all_input_nodes:
                if input_node in constant_nodes and node.op != "get_attr":
                    stack_traces.extend(_impl(input_node))
            stack_traces.append(_node_repr(node))
            return stack_traces

        return "; ".join(_impl(node))

    for node in graph.nodes:
        if node in constant_nodes:
            continue
        for input_node in node.all_input_nodes:
            if input_node in constant_nodes and input_node.op != "get_attr":
                assert isinstance(tensor_meta := input_node.meta["tensor_meta"], TensorMetadata)
                graph_module.register_parameter(
                    target := make_folded_constant_name(input_node),
                    torch.nn.Parameter(
                        torch.zeros(size=tensor_meta.shape, dtype=tensor_meta.dtype),
                        requires_grad=False,
                    ),
                )
                with graph.inserting_before(node):
                    populate_tensor_metadata(get_attr := graph.get_attr(target), tensor_meta)
                add_stack_trace(get_attr, make_fake_stack_trace_header(make_stack_traces(input_node)))
                input_node.replace_all_uses_with(get_attr)

    clean_up_graph_after_modifications(graph_module)


def add_stack_trace(node: Node, msg: str, prepend: bool = False) -> None:
    if node.stack_trace is None:
        node.stack_trace = msg
        return
    if prepend:
        node.stack_trace = f"{msg}, {node.stack_trace}"
    else:
        node.stack_trace = f"{node.stack_trace}, {msg}"


def make_fake_stack_trace_header(code: str) -> str:
    return f'File "?", line 0, in ?\n  {code}'
