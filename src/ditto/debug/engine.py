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

# mypy: disable-error-code=misc

import json
import keyword
import re
from collections.abc import Callable
from functools import cache
from inspect import isclass
from types import NoneType, UnionType
from typing import Any, ClassVar, Literal, get_args

import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from loguru import logger
from onnx import TensorProto
from pydantic import Field, model_serializer, model_validator
from typing_extensions import Self

from ..types import DataType, StrictlyTyped
from .constant import make_constant


class EngineComponent(StrictlyTyped):
    """Represents a component of an engine with configurable attributes.

    Attributes:
        model_config (dict): Configuration for the model.
        KEY_MAP (ClassVar[dict[str, str]]): Mapping of key names.
        attributes (dict[str, Any]): Additional attributes for the component.
    """

    model_config = {"extra": "allow"}
    KEY_MAP: ClassVar[dict[str, str]] = {}

    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, values: Any) -> Any:
        """Resolve and validate model fields.

        Args:
            values (Any): Input values to resolve.

        Returns:
            Any: Resolved values.
        """
        attributes: dict[str, Any] | None = None

        def _resolve(name: str, value: Any) -> tuple[str, Any] | None:
            nonlocal attributes
            _name = cls.KEY_MAP.get(name, name)
            # Put all unrecognized keys in attributes
            # pylint: disable-next=unsupported-membership-test
            if _name not in cls.model_fields:
                if attributes is None:
                    attributes = {}
                attributes[_name] = value
                return None
            # pylint: disable-next=unsubscriptable-object
            _type = cls.model_fields[_name].annotation
            if (  # handling type annotation of the form `T | None`
                isinstance(_type, UnionType) and len(_type_args := get_args(_type)) == 2 and _type_args[1] is NoneType
            ):
                _type = get_args(_type)[0]
            if isinstance(value, str) and isinstance(  # handling Enum-like types in tensorrt
                members := getattr(_type, "__members__", None), dict
            ):
                if value == "BFloat16":
                    value = "Bf16"
                elif value == "N/A due to dynamic shapes":
                    return _name, None
                return _name, members[value.upper()]
            if _type is trt.Dims and isinstance(value, list):
                return _name, trt.Dims(value)
            return _name, value

        if isinstance(values, dict):
            values = dict(item for name, value in values.items() if (item := _resolve(name, value)))
            if attributes is not None:
                values["attributes"] = attributes
        return values

    @model_serializer(mode="wrap")
    def serialize_model(self, original_serializer: Callable[[Self], dict[str, Any]]) -> dict[str, Any]:
        """Serialize the model to a dictionary.

        Args:
            original_serializer (Callable[[Self], dict[str, Any]]): Original serializer function.

        Returns:
            dict[str, Any]: Serialized model data.
        """
        data = original_serializer(self)
        attributes = data.pop("attributes", None)
        if isinstance(attributes, dict):
            data.update(attributes)
        for k, v in data.items():
            if isinstance(v, trt.Dims):
                data[k] = [*v]
                continue
            if isinstance(getattr(type(v), "__members__", None), dict) and hasattr(v, "name"):
                data[k] = v.name.lower()
                continue
        return data


class NamedEngineComponent(EngineComponent):
    """Represents a named engine component.

    Attributes:
        name (str): Name of the engine component.
    """

    name: str

    @property
    def identifier(self) -> str:
        """Identifier for the named engine component."""
        return make_as_identifier(self.name)


class ETensor(NamedEngineComponent):
    """Represents a tensor in the engine.

    Attributes:
        KEY_MAP (ClassVar[dict[str, str]]): Mapping of key names.
        shape (trt.Dims): Shape of the tensor.
        dtype (trt.DataType | None): Data type of the tensor.
        location (trt.TensorLocation | None): Location of the tensor.
    """

    KEY_MAP = {
        "Name": "name",
        "Format/Datatype": "dtype",
        "Location": "location",
        "Dimensions": "shape",
    }
    shape: trt.Dims
    dtype: trt.DataType | None
    location: trt.TensorLocation | None = None

    @classmethod
    @cache
    def make_none(cls) -> Self:
        """Create a tensor representing 'None'.

        Returns:
            Self: A tensor with 'None' values.
        """
        return cls(
            name="None",
            shape=trt.Dims(),
            dtype=None,
        )

    @classmethod
    def from_tensor(cls, tensor: trt.ITensor | None) -> Self:
        """Create an ETensor instance from a trt.ITensor.

        Args:
            tensor (trt.ITensor | None): Tensor to convert.

        Returns:
            Self: Converted ETensor instance.
        """
        if tensor is None:
            return cls.make_none()
        return cls(
            name=tensor.name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            location=tensor.location,
        )

    def __str__(self) -> str:
        loc = f"@{self.location.name.lower()}" if self.location else ""
        dtype = "unknown" if self.dtype is None else self.dtype.name.lower()
        return f"{self.name}: {dtype}{[*self.shape]}{loc}"


class ELayer(NamedEngineComponent):
    """Represents a layer in the engine.

    Attributes:
        KEY_MAP (ClassVar[dict[str, str]]): Mapping of key names.
        inputs (list[ETensor]): List of input tensors.
        outputs (list[ETensor]): List of output tensors.
        layer_type (str): Type of the layer.
    """

    KEY_MAP = {
        "Name": "name",
        "Inputs": "inputs",
        "Outputs": "outputs",
        "LayerType": "layer_type",
    }
    inputs: list[ETensor]
    outputs: list[ETensor]
    layer_type: str

    @classmethod
    def from_layer(cls, layer: trt.ILayer) -> Self:
        """Create an ELayer instance from a trt.ILayer.

        Args:
            layer (trt.ILayer): Layer to convert.

        Returns:
            Self: Converted ELayer instance.
        """
        attrs: dict[str, Any] = {}

        # set common attributes of ILayer
        for key in dir(layer):
            if not (key.startswith("_") or callable(getattr(layer, key))):
                attrs[key] = str(getattr(layer, key))

        # set attributes of the specific layer type
        if concrete_class := get_concrete_class(layer):
            concrete_name = concrete_class.__name__.removeprefix("I").removesuffix("Layer")
            layer_type = f"trt::{concrete_name}"
            layer.__class__ = concrete_class
        else:
            layer_type = layer.type.name.lower()

        base_class_attrs = dir(trt.ILayer)
        for key in dir(layer):
            if key in base_class_attrs and key != "type":
                continue
            if key == "type" and not isinstance(layer.type, trt.LayerType):
                # Required for trt.IActivationLayer
                attrs["algo-type"] = str(layer.type)
                continue
            attr = getattr(layer, key)
            if isinstance(attr, trt.IPluginV2):
                attrs[key] = [
                    f"namespace: {attr.plugin_namespace}",
                    f"type: {attr.plugin_type}",
                    f"version: {attr.plugin_version}",
                ]
            else:
                attrs[key] = str(attr)

        return cls(
            name=layer.name,
            layer_type=layer_type,
            inputs=[ETensor.from_tensor(layer.get_input(i)) for i in range(layer.num_inputs)],
            outputs=[ETensor.from_tensor(layer.get_output(i)) for i in range(layer.num_outputs)],
            attributes=attrs,
        )


class EngineInfo(EngineComponent):
    """Represents information about the engine.

    Attributes:
        KEY_MAP (ClassVar[dict[str, str]]): Mapping of key names.
        bindings (list[str]): List of binding names.
        layers (list[ELayer]): List of layers in the engine.
    """

    KEY_MAP = {
        "Bindings": "bindings",
        "Layers": "layers",
    }
    bindings: list[str]
    layers: list[ELayer]

    @classmethod
    def from_network_definition(cls, network: trt.INetworkDefinition) -> Self:
        """Create an EngineInfo instance from a network definition.

        Args:
            network (trt.INetworkDefinition): Network definition to convert.

        Returns:
            Self: Converted EngineInfo instance.
        """
        return cls(
            bindings=[
                *(network.get_input(i).name for i in range(network.num_inputs)),
                *(network.get_output(i).name for i in range(network.num_outputs)),
            ],
            layers=[ELayer.from_layer(network.get_layer(i)) for i in range(network.num_layers)],
        )

    # pylint: disable-next=too-many-locals
    def as_onnx(
        self,
        profiles: list[trt.IOptimizationProfile] | None = None,
        network_flags: dict[str, bool] | None = None,
    ) -> onnx.ModelProto:
        """Convert engine information to an ONNX model.

        Args:
            profiles (list[trt.IOptimizationProfile] | None): List of optimization profiles. Defaults to None.
            network_flags (dict[str, bool] | None): Network flags. Defaults to None.

        Returns:
            onnx.ModelProto: Converted ONNX model.
        """
        nodes: dict[str, gs.Node] = {}
        degenerate_layers: dict[str, ELayer] = {}
        tensors: dict[str, gs.Constant | gs.Variable] = {}
        redefinition_counts: dict[str, int] = {}
        orphan_tensors: list[ETensor] = []
        input_names: list[str] = []
        output_names: list[str] = []

        def get_tensor_key(t: ETensor) -> str:
            if (count := redefinition_counts.get(t.name, 0)) == 0:
                return t.name
            return t.name + count * " "

        def get_tensor(t: ETensor) -> gs.Constant | gs.Variable:
            if (key := get_tensor_key(t)) not in tensors:
                orphan_tensors.append(t)
                add_as_constant(t)
            return tensors[key]

        def add_as_variable(t: ETensor, *, category: Literal["activation", "input", "output"] = "activation") -> None:
            if t.name in tensors and category == "activation":
                redefinition_counts[t.name] = redefinition_counts.get(t.name, 0) + 1
            name = get_tensor_key(t)
            tensors[name] = gs.Variable(
                name=name,
                dtype=None if t.dtype is None else DataType(t.dtype).to(TensorProto.DataType),
                shape=(*t.shape,),
            )
            if t.name in redefinition_counts:
                return
            if category == "input":
                assert t.name not in input_names
                input_names.append(t.name)
            elif category == "output":
                assert t.name not in output_names
                output_names.append(t.name)

        def add_as_constant(t: ETensor) -> None:
            if t.name in tensors:
                redefinition_counts[t.name] = count = redefinition_counts.get(t.name, 0) + 1
                logger.trace(f"Redefined {count} times: {t}")
            name = get_tensor_key(t)
            if t.dtype is None:
                tensors[name] = gs.Variable(
                    name=name,
                    dtype=None,
                    shape=(*t.shape,),
                )
            else:
                tensors[name] = make_constant(
                    name,
                    shape=(*t.shape,),
                    dtype=DataType(t.dtype).to(torch.dtype),
                )

        def add_as_node(l: ELayer) -> None:  # noqa: E741
            assert l.name not in nodes

            if len(l.outputs) == 0:
                degenerate_layers[l.name] = l
                return

            if l.layer_type.lower() == "constant":
                assert len(l.outputs) == 1
                add_as_constant(l.outputs[0])
                return

            for t in l.inputs:
                if t.name in self.bindings and t.name not in tensors:
                    add_as_variable(t, category="input")

            for t in l.outputs:
                add_as_variable(t, category="output" if t.name in self.bindings else "activation")

            nodes[l.name] = gs.Node(
                op=l.layer_type,
                name=l.name,
                attrs=l.attributes,
                inputs=[get_tensor(t) for t in l.inputs],
                outputs=[get_tensor(t) for t in l.outputs],
            )

        for layer in self.layers:
            logger.trace(f"Adding {layer} as a node")
            add_as_node(layer)

        def json_dumps(obj: Any) -> str:
            return json.dumps(obj, indent="··", sort_keys=True)

        input_ranges = get_shape_ranges(self.bindings, profiles) if profiles is not None else None
        doc_string = "\n".join(
            f"{padstr(section)}\n{contents}"
            for section, contents in {
                "Network Flags": json_dumps(network_flags),
                "Input Ranges": json_dumps(input_ranges),
                "Degenerate Layers": json_dumps(
                    {name: layer.model_dump() for name, layer in degenerate_layers.items()}
                ),
                "Redefined Nodes": json_dumps(redefinition_counts),
                "Orphan Tensors": json_dumps([t.model_dump() for t in orphan_tensors]),
                "": "",
            }.items()
        )

        graph = gs.Graph(
            nodes=list(nodes.values()),
            inputs=[tensors[name] for name in input_names],
            outputs=[tensors[name] for name in output_names],
            doc_string=doc_string,
        )
        graph.cleanup()
        graph.toposort()
        proto = gs.export_onnx(graph, do_type_check=False)
        return proto


def make_as_identifier(s: str) -> str:
    """Generate a valid Python identifier from a string.

    Args:
        s (str): Input string.

    Returns:
        str: Valid Python identifier.
    """
    # Remove invalid characters and replace them with underscores
    s = re.sub(r"\W|^(?=\d)", "_", s)

    # If the result is a Python keyword, add a suffix to make it a valid identifier
    if keyword.iskeyword(s):
        s += "_id"

    return s


def padstr(msg: str, token: str = "-", length: int = 40) -> str:
    """Generate a string msg padded by token at both side, length is adjusted to screen width if not given.

    Args:
        msg (str): message
        token (str): token to pad, must be single character. Defaults to "-".
        length (int): the length of the padded result. Defaults to 40.

    Returns:
        str: padded string
    """
    assert len(token) == 1
    if (pads := length - len(msg)) <= 0:
        return msg
    lpad = pads // 2
    rpad = pads - lpad
    return f"{token * lpad}{msg}{token * rpad}"


ShapeRanges = dict[str, dict[Literal["min", "opt", "max"], tuple[int, ...]]]


def get_shape_ranges(
    input_names: list[str],
    optimization_profiles: list[trt.IOptimizationProfile],
) -> list[ShapeRanges]:
    """Get shape ranges for input names from optimization profiles.

    Args:
        input_names (list[str]): List of input names.
        optimization_profiles (list[trt.IOptimizationProfile]): List of optimization profiles.

    Returns:
        list[ShapeRanges]: List of shape ranges for each profile.
    """
    return [
        {
            name: dict(
                zip(
                    ("min", "opt", "max"),
                    ((*dims,) for dims in profile.get_shape(name)),
                )
            )
            for name in input_names
            if name not in ("logits", "hidden_states_output")
        }
        for profile in optimization_profiles
    ]


def get_concrete_class(layer: trt.ILayer) -> type[trt.ILayer] | None:
    """Get the class of the layer type.

    Args:
        layer (trt.ILayer): Layer to get the class for.

    Returns:
        type[trt.ILayer] | None: Class of the layer type, or None if not found.
    """
    trt_layer_type_to_class = get_trt_layer_type_to_class()
    if (layer_type_name := layer.type.name) in trt_layer_type_to_class:
        return trt_layer_type_to_class[layer_type_name]
    return None


@cache
def get_trt_layer_type_to_class() -> dict[str, type[trt.ILayer]]:
    """Get the mapping from layer type name to the corresponding trt.ILayer subclass.

    Returns:
        dict[str, type[trt.ILayer]]: Mapping from layer type name to trt.ILayer subclass.
    """
    trt_layer_classes = {
        name.lower().removeprefix("i").removesuffix("layer"): value
        for name, value in vars(trt).items()
        if isclass(value) and value is not trt.ILayer and issubclass(value, trt.ILayer)
    }
    return {
        name: trt_layer_classes[lower_cased_name]
        for name in trt.LayerType.__members__
        if (lower_cased_name := name.lower().replace("_", "")) in trt_layer_classes
    }
