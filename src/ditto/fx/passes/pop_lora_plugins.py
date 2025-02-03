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

import operator
import random
from collections.abc import Iterable
from typing import get_args

import tensorrt as trt
import torch
from loguru import logger
from peft import LoraConfig
from pydantic import Field
from torch.fx import GraphModule, Node

from ...arguments import TRTLLMArgumentHint
from ...constants import ATTN_GATED_MLP_PREFIXES, ATTN_QKV_PREFIXES
from ...contexts import temporary_random_seed
from ...literals import LoraCheckpointLiteral, LoraPluginInputPrefix
from ...types import DataType, verify
from ..metadata_keys import LORA_PROTOS
from ..nodes import AddTensor, Cat
from ..subgraphs import Linear, LoraProto
from ..targets import LoraPlugin, LoraPluginInputs
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class PopLoraPlugins(NodewiseOptimizationPass):
    """Pass that converts LoRA subgraphs into TensorRT-LLM LoRA plugins."""

    argument_hint: TRTLLMArgumentHint
    state_dicts: dict[int, dict[str, torch.Tensor]] = Field(default_factory=dict)
    target_modules: dict[int, set[LoraPluginInputPrefix]] = Field(default_factory=dict)
    max_rank: int = 0
    ckpt_source: LoraCheckpointLiteral = "hf"

    @property
    def ckpt_dir(self) -> list[str]:
        """List of checkpoint directory paths for each LoRA task."""
        return [f"lora/{task_uid}" for task_uid in self.state_dicts]

    def update_state_dicts(
        self,
        proto_map: dict[LoraPluginInputPrefix, LoraProto],
        layer_index: int,
        model_name: str | None = None,
    ) -> None:
        """Update the combined state dictionaries with each of the LoRA parameters.

        Args:
            proto_map (dict[LoraPluginInputPrefix, LoraProto]): Mapping of LoRA prefixes to their prototypes
            layer_index (int): Index of the current layer
            model_name (str | None): Optional name of the model
        """
        for lora_prefix, proto in proto_map.items():
            for lora_task_uid, state_dict in proto.state_dicts.items():
                layer_prefix = create_random_layer_prefix(lora_task_uid, model_name)
                module_prefix = create_random_module_prefix(lora_task_uid, model_name)
                global_state_dict = self.state_dicts.setdefault(lora_task_uid, {})
                global_state_dict.update(
                    {
                        f"{layer_prefix}.{layer_index}.{module_prefix}.{lora_prefix}.{key}": value
                        for key, value in state_dict.items()
                    }
                )
                target_modules = self.target_modules.setdefault(lora_task_uid, set())
                target_modules.add(lora_prefix)
            self.max_rank = max(self.max_rank, proto.max_low_rank)

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (linear := Linear.configure_from(node))
            # Metadata "layer_index" is populated by `IndexLayers`
            and (layer_index := verify(linear.mm.meta.get("layer_index", None), as_type=int)) is not None
            and (
                # Metadata "lora_protos" are populated by `StashLoraSubgraphs` in advance
                lora_protos := verify(
                    linear.mm.meta.get(LORA_PROTOS, {}), as_type=dict[LoraPluginInputPrefix, LoraProto]
                )
            )
        ):
            return {}

        try:
            lora_plugin, prefixes = create_lora_plugin(lora_protos, linear.out_features)
        except RuntimeError as e:
            logger.error(f"Failed to create Lora plugin: {e}")
            return {}

        graph = node.graph
        model_name = None if graph.owning_module is None else type(graph.owning_module).__name__
        self.update_state_dicts(lora_protos, layer_index, model_name)
        if (
            lora_plugin_inputs := LoraPluginInputs.create_and_sync(
                graph,
                self.argument_hint,
                prefixes,
                layer_index,
            )
        ) is None:
            return {}

        last_node = linear.input_node
        with graph.inserting_after(last_node):
            last_node = lora_plugin_node = graph.call_function(
                lora_plugin,
                (last_node,),
                lora_plugin_inputs.model_dump(),
            )

        if (num_outputs := lora_plugin.num_lora_modules) > 1:
            outputs: list[Node] = []
            for i in range(num_outputs):
                with graph.inserting_after(last_node):
                    outputs.append(
                        last_node := graph.call_function(
                            operator.getitem,
                            (lora_plugin_node, i),
                        )
                    )
            with graph.inserting_after(outputs[-1]):
                last_node = Cat.create(graph, outputs, -1).node

        with graph.inserting_after(max(linear.output_node, last_node)):
            add_node = AddTensor.create(graph, linear.output_node, last_node).node

        return {
            linear.output_node: ReplaceAllUses(
                by=add_node,
                # required when `num_outputs` == 1
                replace_user_only_if=lambda n: n is not add_node,
            )
        }

    def postprocess(self, graph_module: GraphModule) -> None:
        super().postprocess(graph_module)
        if self.max_rank == 0:
            return
        graph_module.meta["lora_state_dicts"] = self.state_dicts
        all_target_modules = set().union(*self.target_modules.values())
        if any(prefix in all_target_modules for prefix in ATTN_QKV_PREFIXES):
            all_target_modules.update(ATTN_QKV_PREFIXES)
        sorted_target_modules = sort_lora_prefixes(all_target_modules)
        graph_module.meta["lora_config"] = {
            "lora_dir": self.ckpt_dir,
            "lora_ckpt_source": self.ckpt_source,
            "max_lora_rank": self.max_rank,
            "lora_target_modules": sorted_target_modules,
            "trtllm_modules_to_hf_modules": {m: m for m in sorted_target_modules},
        }
        if not (peft_configs := verify(graph_module.meta.get("peft_configs"), as_type=dict[int, LoraConfig])):
            return
        for lora_task_uid, target_modules in self.target_modules.items():
            if lora_task_uid in peft_configs:
                peft_configs[lora_task_uid].target_modules = sort_lora_prefixes(target_modules)


def create_lora_plugin(
    proto_map: dict[LoraPluginInputPrefix, LoraProto],
    out_features: int,
) -> tuple[LoraPlugin, tuple[LoraPluginInputPrefix, ...]]:
    """Create a LoRA plugin from a map of LoRA prototypes.

    Args:
        proto_map (dict[LoraPluginInputPrefix, LoraProto]): Mapping of LoRA prefixes to their prototypes
        out_features (int): Number of output features for the linear layer

    Returns:
        tuple[LoraPlugin, tuple[LoraPluginInputPrefix, ...]]: A tuple containing the created LoRA plugin
            and a tuple of LoRA prefixes

    Raises:
        RuntimeError: If output features don't match the sum of LoRA adapter output features
    """
    out_hidden_size_map: dict[LoraPluginInputPrefix, int] = {
        prefix: proto.out_hidden_size for prefix, proto in proto_map.items()
    }
    for prefix_group in (ATTN_QKV_PREFIXES, ATTN_GATED_MLP_PREFIXES):
        out_hidden_size_map = adjust_out_hidden_size_map(out_hidden_size_map, out_features, prefix_group)

    if out_features != (total_out_hidden_size := sum(out_hidden_size_map.values())):
        raise RuntimeError(
            f"The number of output features of the linear layer ({out_features}) does not match "
            f"the sum of the output features of the Lora adapters ({total_out_hidden_size})."
        )

    first_proto = next(iter(proto_map.values()))
    return LoraPlugin(
        in_hidden_size=first_proto.in_hidden_size,
        transa=first_proto.transa,
        transb=first_proto.transb,
        type_id=DataType(first_proto.dtype).to(trt.DataType),
        remove_input_padding=first_proto.remove_input_padding,
        max_low_rank=max(proto.max_low_rank for proto in proto_map.values()),
        weight_index=first_proto.weight_index,
        out_hidden_sizes=list(out_hidden_size_map.values()),
    ), tuple(out_hidden_size_map.keys())


def adjust_out_hidden_size_map(
    out_hidden_size_map: dict[LoraPluginInputPrefix, int],
    out_features: int,
    prefix_group: tuple[LoraPluginInputPrefix, ...],
) -> dict[LoraPluginInputPrefix, int]:
    """Adjust the output hidden size map to account for missing prefixes.

    Args:
        out_hidden_size_map (dict[LoraPluginInputPrefix, int]): Current mapping of prefixes to output sizes
        out_features (int): Total number of output features
        prefix_group (tuple[LoraPluginInputPrefix, ...]): Group of prefixes to check for missing entries

    Returns:
        dict[LoraPluginInputPrefix, int]: Updated mapping with missing prefixes added
    """
    if all(prefix in prefix_group for prefix in out_hidden_size_map.keys()) and (
        missing_prefixes := [prefix for prefix in prefix_group if prefix not in out_hidden_size_map]
    ):
        missing_out_features = out_features - sum(out_hidden_size_map.values())
        num_missing_prefixes = len(missing_prefixes)
        return {
            prefix: out_hidden_size_map.get(prefix, missing_out_features // num_missing_prefixes)
            for prefix in prefix_group
        }
    return out_hidden_size_map


def sort_lora_prefixes(prefixes: Iterable[LoraPluginInputPrefix]) -> list[LoraPluginInputPrefix]:
    """Sort LoRA prefixes according to their order in the LoraPluginInputPrefix enum.

    Args:
        prefixes (Iterable[LoraPluginInputPrefix]): Iterable of LoRA prefixes to sort

    Returns:
        list[LoraPluginInputPrefix]: Sorted list of LoRA prefixes
    """
    all_lora_prefixes: tuple[LoraPluginInputPrefix, ...] = get_args(LoraPluginInputPrefix)
    return sorted(prefixes, key=all_lora_prefixes.index)


def create_random_layer_prefix(*seeds: int | str | bytes | None) -> str:
    """Create a random layer prefix from food-related words.

    Args:
        *seeds: Seeds for random number generation

    Returns:
        str: A randomly generated layer prefix string
    """
    ingredients = [
        "spinach",
        "beef",
        "chicken",
        "tofu",
        "mushroom",
        "carrot",
        "potato",
        "tomato",
        "cheese",
        "salmon",
        "shrimp",
        "eggplant",
        "zucchini",
        "pork",
        "tuna",
        "broccoli",
        "cauliflower",
        "lamb",
        "chickpea",
        "lentil",
        "asparagus",
        "avocado",
        "cabbage",
        "celery",
        "cucumber",
        "garlic",
        "kale",
        "lettuce",
        "onion",
        "pepper",
        "quinoa",
        "radish",
        "squash",
        "sweet_potato",
        "tempeh",
        "turkey",
        "watercress",
        "yam",
        "beans",
        "corn",
    ]
    dishes = [
        "burger",
        "pasta",
        "curry",
        "stew",
        "salad",
        "soup",
        "sandwich",
        "pizza",
        "taco",
        "stir_fry",
        "casserole",
        "pie",
        "roll",
        "dumpling",
        "noodles",
        "risotto",
        "lasagna",
        "quiche",
        "frittata",
        "bowl",
        "burrito",
        "chili",
        "fajita",
        "gnocchi",
        "gyoza",
        "kebab",
        "meatball",
        "omelette",
        "paella",
        "pho",
        "pilaf",
        "quesadilla",
        "ravioli",
        "sushi",
        "tempura",
        "teriyaki",
        "udon",
        "wrap",
        "yakitori",
        "ziti",
    ]
    with temporary_random_seed(*seeds):
        return f"{random.choice(ingredients)}.{random.choice(dishes)}"


def create_random_module_prefix(*seeds: int | str | bytes | None) -> str:
    """Create a random module prefix from animal names.

    Args:
        *seeds: Seeds for random number generation

    Returns:
        str: A randomly generated module prefix string
    """
    animals = [
        "lion",
        "tiger",
        "elephant",
        "giraffe",
        "zebra",
        "kangaroo",
        "panda",
        "cheetah",
        "dolphin",
        "penguin",
        "bear",
        "wolf",
        "fox",
        "rabbit",
        "deer",
        "monkey",
        "otter",
        "hedgehog",
        "eagle",
        "owl",
    ]
    with temporary_random_seed(*seeds):
        return random.choice(animals)
