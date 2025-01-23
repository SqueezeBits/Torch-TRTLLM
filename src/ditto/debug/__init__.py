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

from .engine import EngineInfo
from .fx import build_onnx_from_fx
from .io import (
    get_debug_artifacts_dir,
    open_debug_artifact,
    save_for_debug,
    save_onnx_without_weights,
    should_save_debug_artifacts,
)
from .memory import get_device_memory_footprint, get_host_memory_footprint, get_memory_footprint
from .network import builder_config_as_dict, get_human_readable_flags
from .plugin import enable_plugin_debug_info_hook, plugin_debug_info_hook
