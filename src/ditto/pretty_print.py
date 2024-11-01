import contextlib
import re
from collections.abc import Generator
from typing import Any

import tensorrt as trt
import torch
from torch.fx.experimental.sym_node import SymNode


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


def get_network_ir(network: trt.INetworkDefinition) -> str:
    lines: list[str] = []
    for i in range(network.num_inputs):
        lines.append(f"{get_tensor_repr(network.get_input(i))} [network input]")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        inputs = ", ".join(get_alias(layer.get_input(j)) for j in range(layer.num_inputs))
        outputs = ", ".join(get_tensor_repr(layer.get_output(j)) for j in range(layer.num_outputs))
        lines.append(f"{outputs} = {layer.type.name}({inputs})")
    for i in range(network.num_outputs):
        lines.append(f"{get_tensor_repr(network.get_output(i))} [network output]")
    return "\n".join(lines)


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
