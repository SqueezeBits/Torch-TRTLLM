import contextlib
import keyword
import operator
import re
from collections.abc import Callable, Generator
from typing import Any

import tensorrt as trt
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.experimental.sym_node import SymNode
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tensorrt import dtype


@contextlib.contextmanager
def brief_tensor_repr() -> Generator[None, None, None]:
    def tensor_repr(self: torch.Tensor, *, tensor_contents=None) -> str:
        dtype_repr = f"{self.dtype}".removeprefix("torch.")
        return f"Tensor(shape={tuple(self.shape)}, dtype={dtype_repr}, device={self.device})"

    original_tensor__repr__ = torch.Tensor.__repr__
    torch.Tensor.__repr__ = tensor_repr
    try:
        yield None
    finally:
        torch.Tensor.__repr__ = original_tensor__repr__


@contextlib.contextmanager
def detailed_sym_node_str() -> Generator[None, None, None]:
    def sym_node_str(self: SymNode) -> str:
        return f"{self._expr}({self.expr})"

    original_sym_node_str = SymNode.str
    SymNode.str = sym_node_str
    try:
        yield None
    finally:
        SymNode.str = original_sym_node_str


def reformat_unnamed_layer(input_string: str) -> str:
    """E.g. "(Unnamed Layer* 1095) [Slice]" -> "slice_1095"."""
    # Use regex to capture the number and the part after the bracket
    match = re.search(r"\* (\d+)\) \[(\w+)\]", input_string)
    if match:
        # Extract the number and the word in brackets
        number = match.group(1)
        word = match.group(2).lower()  # Convert the word to lowercase
        # Return the formatted string
        return f"{word}_{number}"
    return input_string[:]


def get_alias(tensor: trt.ITensor | None) -> str:
    if tensor is None:
        return "None"

    def _simplify(name: str) -> str:
        if (index := name.find("]_output")) != -1:
            name = name[: index + 1]
        name = name.split("-")[-1]
        if name.startswith("[") and name.endswith("]"):
            name = name.removeprefix("[").removesuffix("]")
        name = reformat_unnamed_layer(name)
        return name

    long_name = tensor.name
    while (short_name := _simplify(long_name)) != long_name:
        long_name = short_name
    return short_name


def get_tensor_repr(tensor: trt.ITensor | None) -> str:
    if tensor is None:
        return "None"
    name = get_alias(tensor)
    dtype_ = tensor.dtype.name
    shape = tensor.shape
    device = tensor.location.name
    return f"{name}: {dtype_}{shape}@{device}"


def get_network_ir(
    network: trt.INetworkDefinition,
    profiles: list[trt.IOptimizationProfile] | None = None,
) -> str:
    return "\n".join(
        [
            get_dynamic_input_ranges(network, profiles) if profiles else "",
            build_fake_graph_module(network).print_readable(print_output=False),
        ]
    )


def build_fake_graph_module(network: trt.INetworkDefinition) -> GraphModule:
    graph = Graph()
    nodes: dict[str, Node] = {}

    def extract_tensor_metadata(t: trt.ITensor) -> TensorMetadata:
        return TensorMetadata(
            shape=torch.Size(t.shape[:]),
            dtype=dtype._from(t.dtype).to(torch.dtype),  # type: ignore
            requires_grad=False,
            stride=(),
            memory_format=None,
            is_quantized=False,
            qparams={},
        )

    def make_as_identifier(s: str) -> str:
        # Remove invalid characters and replace them with underscores
        s = re.sub(r"\W|^(?=\d)", "_", s)

        # If the result is a Python keyword, add a suffix to make it a valid identifier
        if keyword.iskeyword(s):
            s += "_id"

        return s

    def make_fake_stack_trace(layer_name: str) -> str:
        return f'File "None", line 0, in <None>\n  {layer_name}'

    fake_targets: dict[str, Callable[..., Any]] = {}

    def get_fake_target(layer: trt.ILayer) -> Callable[..., Any]:
        if (layer_type := layer.type.name.lower()) in fake_targets:
            return fake_targets[layer_type]

        def fake_target(*args, **kwargs):
            ...

        fake_target.__name__ = layer_type
        fake_target.__module__ = "trt"
        fake_targets[layer_type] = fake_target
        return fake_target

    class FakeModule(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.fake_params: dict[str, torch.nn.Parameter] = {}

        def register_fake_parameter(self, name: str, param: torch.nn.Parameter) -> None:
            self.fake_params[name] = param

        def __getattr__(self, name: str) -> Any:
            if name in self.fake_params:
                return self.fake_params[name]
            return super().__getattr__(name)

    fake_module = FakeModule()
    graph_module = GraphModule(FakeModule(), graph)

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
            identifier = make_as_identifier(layer.name)
            if layer.type == trt.LayerType.CONSTANT:
                fake_module.register_fake_parameter(
                    identifier,
                    torch.nn.Parameter(
                        torch.zeros(size=tensor_meta.shape, dtype=tensor_meta.dtype),
                        requires_grad=False,
                    ),
                )
            output_node = nodes[outputs[0].name] = (
                graph.get_attr(identifier)
                if layer.type == trt.LayerType.CONSTANT
                else graph.call_function(
                    fake_target,
                    tuple(None if x is None else nodes[x.name] for x in inputs),
                )
            )
            output_node.meta["tensor_meta"] = tensor_meta
            output_node.stack_trace = make_fake_stack_trace(layer.name)
        else:
            outputs_node = nodes[f"{layer.name}_output"] = graph.call_function(
                fake_target,
                tuple(nodes[x.name] for x in inputs),
            )
            for i, output in enumerate(outputs):
                output_node = nodes[output.name] = graph.call_function(
                    operator.getitem,
                    (outputs_node, i),
                )
                output_node.meta["tensor_meta"] = extract_tensor_metadata(output)
                output_node.stack_trace = make_fake_stack_trace(layer.name)
    graph.output(tuple(nodes[network.get_output(i).name] for i in range(network.num_outputs)))
    for node in graph.nodes:
        if stack_trace := node.stack_trace:
            node.stack_trace = f"{stack_trace}, users: {len(node.users)}"
    return graph_module


def get_dynamic_input_ranges(
    network: trt.INetworkDefinition,
    optimization_profiles: list[trt.IOptimizationProfile],
) -> str:
    messages = []
    # Loop through all network inputs
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        # Loop through all optimization profiles in the builder configuration
        for profile_index, profile in enumerate(optimization_profiles):
            # Get the min, opt, and max shape for this input
            min_shape, opt_shape, max_shape = profile.get_shape(input_tensor.name)
            messages.append(f"# Profile {profile_index} for '{input_tensor.name}':")
            messages.append(f"#   Min shape: {min_shape}")
            messages.append(f"#   Opt shape: {opt_shape}")
            messages.append(f"#   Max shape: {max_shape}")
    return "\n".join(messages)


def builder_config_as_dict(builder_config: trt.IBuilderConfig) -> dict[str, Any]:
    """Save all attributes of a TensorRT IBuilderConfig object as a JSON file.

    Args:
        builder_config (trt.IBuilderConfig): The TensorRT builder configuration to save.
        file_path (str): The path to the JSON file where the configuration will be saved.
    """
    # Dictionary to store attribute values
    config_data: dict[str, Any] = {}

    T = bool | float | int | str | None  # noqa: N806

    def normalize(value: Any) -> T | list[Any] | tuple[Any, ...] | dict[str, Any]:
        value = getattr(value, "name", value)
        if isinstance(value, list):
            return [normalize(x) for x in value]
        if isinstance(value, tuple):
            return tuple(normalize(x) for x in value)
        if isinstance(value, dict):
            return {f"{k}": normalize(v) for k, v in value.items()}
        if not isinstance(value, T):
            return f"{value}"
        return value

    def bitmask_to_bool_list(bitmask: int) -> list[bool]:
        # Ensure the bitmask is within the int32 range
        if bitmask < 0 or bitmask > 0xFFFFFFFF:
            raise ValueError("Bitmask should be a 32-bit integer")

        # Convert the bitmask to a list of booleans
        return [(bitmask & (1 << i)) != 0 for i in range(32)]

    # Loop through attributes in IBuilderConfig and retrieve their values
    for attr in dir(builder_config):
        # Filter out private attributes, methods, and unsupported types
        if not attr.startswith("_") and not callable(getattr(builder_config, attr)):
            try:
                # Retrieve attribute value
                value = getattr(builder_config, attr)
                if attr == "flags" and isinstance(value, int):
                    value = {
                        f"{enum_value.value:02d}:{name}": flag
                        for (name, enum_value), flag in zip(
                            trt.BuilderFlag.__members__.items(), bitmask_to_bool_list(value)
                        )
                    }
                config_data[attr] = normalize(value)
            except Exception as e:
                # Handle any errors in retrieving attribute value
                config_data[attr] = f"Error retrieving: {str(e)}"

    return config_data
