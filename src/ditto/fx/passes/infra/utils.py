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

from torch.fx import Graph
from transformers import PretrainedConfig

from ....types import verify


def get_pretrained_config(graph: Graph) -> PretrainedConfig | None:
    """Get the pretrained config from a graph's owning module metadata.

    Args:
        graph (Graph): The FX graph to get the config from

    Returns:
        PretrainedConfig | None: The pretrained config if found and valid, otherwise None
    """
    pretrained_config: PretrainedConfig | None = (
        verify(
            graph_module.meta.get("pretrained_config"),
            as_type=PretrainedConfig,
        )
        if (graph_module := graph.owning_module)
        else None
    )
    return pretrained_config
