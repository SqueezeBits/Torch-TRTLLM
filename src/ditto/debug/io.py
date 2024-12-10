import io
import json
import os
from contextlib import nullcontext
from functools import cache
from typing import IO, Any, overload

import onnx
import tensorrt as trt
from loguru import logger
from onnx.external_data_helper import _get_all_tensors, uses_external_data
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch.fx.graph import _PyTreeCodeGen
from torch_tensorrt.logging import TRT_LOGGER

from ..constants import DEBUG_ARTIFACTS_DIR, DEFAULT_ONNX_PROTO_SIZE_THRESHOLD
from ..pretty_print import detailed_sym_node_str
from .engine import EngineInfo
from .fx import build_onnx_from_fx
from .network import get_human_readable_flags


@cache
def get_debug_artifacts_dir() -> str | None:
    if DEBUG_ARTIFACTS_DIR is None:
        return None
    os.makedirs(DEBUG_ARTIFACTS_DIR, exist_ok=True)
    logger.info(f"DEBUG_ARTIFACTS_DIR: {DEBUG_ARTIFACTS_DIR}")
    return DEBUG_ARTIFACTS_DIR


def should_save_debug_artifacts() -> bool:
    return get_debug_artifacts_dir() is not None


def open_debug_artifact(filename: str, mode: str = "w") -> IO[Any] | nullcontext[None]:
    if d := get_debug_artifacts_dir():
        path = os.path.join(d, filename)
        actions = {
            "w": "Writing",
            "a": "Appending",
            "r": "Reading",
        }
        for k, v in actions.items():
            if k in mode:
                action = v
                break
        else:
            action = "Opening"
        logger.info(f"{action} debug artifact at {path}")
        return open(path, mode)
    return nullcontext(None)


@overload
def save_for_debug(
    name: str,
    item: ExportedProgram | GraphModule,
) -> None:
    ...


@overload
def save_for_debug(
    name: str,
    item: trt.INetworkDefinition | trt.ICudaEngine | trt.IHostMemory,
    profiles: list[trt.IOptimizationProfile] | None = None,
) -> None:
    ...


def save_for_debug(
    name: str,
    item: ExportedProgram | GraphModule | trt.INetworkDefinition | trt.ICudaEngine | trt.IHostMemory,
    profiles: list[trt.IOptimizationProfile] | None = None,
) -> None:
    if isinstance(item, ExportedProgram):
        save_exported_program_for_debug(name, item)
        return

    if isinstance(item, GraphModule):
        save_graph_module_for_debug(name, item)
        return

    if isinstance(item, trt.INetworkDefinition):
        save_network_for_debug(name, item, profiles=profiles)
        return

    if isinstance(item, trt.ICudaEngine | trt.IHostMemory):
        save_engine_for_debug(name, item, profiles=profiles)
        return


def save_network_for_debug(
    name: str,
    network: trt.INetworkDefinition,
    profiles: list[trt.IOptimizationProfile] | None = None,
) -> None:
    flags = get_human_readable_flags(network)
    engine_info = EngineInfo.from_network_definition(network)
    _save_engine_info(name, engine_info, profiles=profiles, network_flags=flags)


def save_engine_for_debug(
    name: str,
    engine: trt.ICudaEngine | trt.IHostMemory,
    profiles: list[trt.IOptimizationProfile] | None = None,
) -> None:
    with open_debug_artifact(f"{name}.json") as f:
        if f:
            if isinstance(engine, trt.IHostMemory):
                with io.BytesIO() as engine_bytes:
                    engine_bytes.write(engine)
                    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_bytes.getvalue())
            inspector = engine.create_engine_inspector()
            engine_info_dict = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            json.dump(engine_info_dict, f, indent=2, sort_keys=True)
            _save_engine_info(name, EngineInfo.model_validate(engine_info_dict), profiles=profiles)


def _save_engine_info(
    name: str,
    engine_info: EngineInfo,
    profiles: list[trt.IOptimizationProfile] | None = None,
    network_flags: dict[str, bool] | None = None,
) -> None:
    with open_debug_artifact(f"{name}.onnx", "wb") as f:
        if f:
            save_onnx_without_weights(engine_info.as_onnx(profiles, network_flags), f)


def save_exported_program_for_debug(
    name: str,
    exported_program: ExportedProgram,
) -> None:
    save_graph_module_for_debug(name, exported_program.graph_module)


def save_graph_module_for_debug(
    name: str,
    graph_module: GraphModule,
) -> None:
    with (
        detailed_sym_node_str(),
        open_debug_artifact(f"{name}.py") as code_file,
        open_debug_artifact(f"{name}.mlir") as graph_file,
        open_debug_artifact(f"{name}.onnx", "wb") as onnx_file,
    ):
        if code_file and graph_file and onnx_file:
            import_statements = ["import torch"]
            if isinstance(graph_module.graph._codegen, _PyTreeCodeGen):
                import_statements.extend(
                    [
                        "import torch.fx._pytree as fx_pytree",
                        "import torch.utils._pytree as pytree",
                    ]
                )
            import_statements.append("\n\n")
            header = "\n".join(import_statements)
            code_file.write(f"{header}{graph_module.print_readable(print_output=False)}")
            graph_file.write(f"{graph_module.graph}")
            save_onnx_without_weights(build_onnx_from_fx(graph_module), onnx_file)


def save_onnx_without_weights(
    proto: onnx.ModelProto,
    f: IO[bytes],
    *,
    size_threshold: int = DEFAULT_ONNX_PROTO_SIZE_THRESHOLD,
    convert_attribute: bool = False,
) -> None:
    onnx.convert_model_to_external_data(
        proto,
        location="null",
        size_threshold=size_threshold,
        convert_attribute=convert_attribute,
    )
    for tensor in _get_all_tensors(proto):
        if uses_external_data(tensor) and tensor.HasField("raw_data"):
            tensor.ClearField("raw_data")
    serialized_proto = onnx._get_serializer(None, f).serialize_proto(proto)
    f.write(serialized_proto)
