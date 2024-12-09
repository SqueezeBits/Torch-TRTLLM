import json
import keyword
import re
import sys
from collections.abc import Callable
from functools import cache
from types import NoneType, UnionType
from typing import Any, ClassVar, Literal, get_args

import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from loguru import logger
from onnx import TensorProto
from onnx.helper import make_tensor
from onnx_graphsurgeon.ir.tensor import LazyValues
from pydantic import model_serializer, model_validator
from typing_extensions import Self

from ..types import DataType, StrictlyTyped


class EngineComponent(StrictlyTyped):
    model_config = {"extra": "allow"}
    KEY_MAP: ClassVar[dict[str, str]] = {}

    attributes: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve(cls, values: Any) -> Any:
        attributes: dict[str, Any] | None = None

        def _resolve(name: str, value: Any) -> tuple[str, Any] | None:
            nonlocal attributes
            _name = cls.KEY_MAP.get(name, name)
            # Put all unrecognized keys in attributes
            if _name not in cls.model_fields:
                if attributes is None:
                    attributes = {}
                attributes[_name] = value
                return None
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
                return _name, members[value.upper()]
            if _type is trt.Dims and isinstance(value, list):
                return _name, trt.Dims(value)
            return _name, value

        if isinstance(values, dict):
            values = dict([item for name, value in values.items() if (item := _resolve(name, value))])
            if attributes is not None:
                values["attributes"] = attributes
        return values

    @model_serializer(mode="wrap")
    def serialize_model(self, original_serializer: Callable[[Self], dict[str, Any]]) -> dict[str, Any]:
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
    name: str

    @property
    def identifier(self) -> str:
        return make_as_identifier(self.name)


class ETensor(NamedEngineComponent):
    KEY_MAP = {
        "Name": "name",
        "Format/Datatype": "dtype",
        "Location": "location",
        "Dimensions": "shape",
    }
    shape: trt.Dims
    dtype: trt.DataType
    location: trt.TensorLocation | None = None

    @classmethod
    @cache
    def get_none(cls) -> Self:
        return cls(
            name="None",
            shape=trt.Dims(),
            dtype=trt.DataType.BOOL,
        )

    @classmethod
    def from_tensor(cls, tensor: trt.ITensor | None) -> Self:
        if tensor is None:
            return cls.get_none()
        return cls(
            name=tensor.name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            location=tensor.location,
        )

    def __str__(self) -> str:
        loc = f"@{self.location.name.lower()}" if self.location else ""
        return f"{self.name}: {self.dtype.name.lower()}{[*self.shape,]}{loc}"


class ELayer(NamedEngineComponent):
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
        return cls(
            name=layer.name,
            layer_type=layer.type.name.lower(),
            inputs=[ETensor.from_tensor(layer.get_input(i)) for i in range(layer.num_inputs)],
            outputs=[ETensor.from_tensor(layer.get_output(i)) for i in range(layer.num_outputs)],
        )


class EngineInfo(EngineComponent):
    KEY_MAP = {
        "Bindings": "bindings",
        "Layers": "layers",
    }
    bindings: list[str]
    layers: list[ELayer]

    @classmethod
    def from_network_definition(cls, network: trt.INetworkDefinition) -> Self:
        return cls(
            bindings=[
                *(network.get_input(i).name for i in range(network.num_inputs)),
                *(network.get_output(i).name for i in range(network.num_outputs)),
            ],
            layers=[ELayer.from_layer(network.get_layer(i)) for i in range(network.num_layers)],
        )

    def as_onnx(
        self,
        profiles: list[trt.IOptimizationProfile] | None = None,
        network_flags: dict[str, bool] | None = None,
    ) -> onnx.ModelProto:
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
                dtype=DataType(t.dtype).to(TensorProto.DataType),
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
            torch_values = torch.zeros((*t.shape,), dtype=DataType(t.dtype).to(torch.dtype))
            if torch_values.dtype != torch.bool:
                byte_vals = torch_values.data_ptr().to_bytes(
                    torch.numel(torch_values) * torch_values.element_size(), sys.byteorder
                )
                onnx_tensor_proto = make_tensor(
                    name=name,
                    data_type=DataType(t.dtype).to(TensorProto.DataType),
                    dims=(*t.shape,),
                    vals=byte_vals,
                    raw=True,
                )
                values = LazyValues(onnx_tensor_proto)
            else:
                values = torch_values.numpy(force=True)
            tensors[name] = gs.Constant(
                name=name,
                values=values,
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
        token (str): token to pad, must be single character
        length (int | None, optional): the length of the padded result. Defaults to None.

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
    logger.info(
        "You can ignore the following error messages 'IOptimizationProfile::getDimensions: Error Code 4: "
        "API Usage Error (...)', which are caused by inputs and outputs without optimization profiles."
    )
    return [
        {
            name: dict(
                zip(
                    ("min", "opt", "max"),
                    ((*dims,) for dims in profile.get_shape(name)),
                )
            )
            for name in input_names
        }
        for profile in optimization_profiles
    ]
