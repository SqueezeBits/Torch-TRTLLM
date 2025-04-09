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

__version__ = "0.2.0"

# Note: logging must be imported first to set up the loguru logging.
from . import logging, constants, patches  # noqa: I001
from .fx import PretrainedConfigGenerationError, generate_trtllm_engine_config
from .convert import convert
from .export import export
from .inline import inline
from .api import trtllm_build, trtllm_export
from .arguments.dynamic_dim import DynamicDimension, DynamicDimensionType
from .arguments.torch_export_arguments import TorchExportArguments
from .configs import *
from .conversion import *
from .contexts import brief_tensor_repr, detailed_sym_node_str
from .peft import load_peft_adapters
from .types import (
    BuiltInConstant,
    DeviceLikeType,
    Number,
    SymbolicInteger,
    SymbolicShape,
    DataType,
)
