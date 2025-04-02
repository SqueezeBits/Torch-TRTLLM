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
# ruff: noqa

# The order of the patches matters.
from .torch import patch_modulelist_getitem
from .transformers import patch_attention_mask_converter_make_causal_mask
from .auto_awq import patch_wqlinear_mm_func_forward
from .auto_gptq import patch_dynamically_import_quantlinear
from .compressed_tensors import patch_decompress_weight

# Do NOT import from .trtllm! We don't want to apply the trtllm patches here.
