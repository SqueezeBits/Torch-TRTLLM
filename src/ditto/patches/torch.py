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

import transformers
from torch.nn.modules import Module, ModuleList

from .patch import custom_patch


@custom_patch(
    name="torch.nn.modules.container.ModuleList.__getitem__",
    reason=(
        "preventing `torch.export` failure observed in transformers>=4.47.0 "
        "by some models at the lines like `for decoder_layer in self.layers[': self.config.num_hidden_layers]:`. "
        "For example, see https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/gemma/modeling_gemma.py#L574"
    ),
    required=transformers.__version__ >= "4.47.0",
    env_var_to_disable="DISABLE_TORCH_MODULELIST_PATCH",
)
def patch_modulelist_getitem() -> None:
    original_modulelist_getitem = ModuleList.__getitem__

    def patched_modulelist_getitem(self: ModuleList, idx: int | slice) -> ModuleList | Module:
        # This patch is required for `torch.export` failure for transformers>=4.47.0
        # at the lines like `for decoder_layer in self.layers[': self.config.num_hidden_layers]:`.
        # For example, see https://github.com/huggingface/transformers/blob/5d7739f15a6e50de416977fe2cc9cb516d67edda/src/transformers/models/llama/modeling_llama.py#L896
        if (
            isinstance(idx, slice)
            and self.__class__.__name__ == "ModuleList"
            and self.__class__.__module__ == "torch.nn.modules.container"
        ):
            # The direct access of `self.__class__` at this line is causing the failure.
            # return self.__class__(list(self._modules.values())[idx])
            return ModuleList(list(self._modules.values())[idx])
        return original_modulelist_getitem(self, idx)

    ModuleList.__getitem__ = patched_modulelist_getitem
