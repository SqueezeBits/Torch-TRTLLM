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

import warnings

from .patch import custom_patch

warnings.simplefilter("ignore", FutureWarning)

from auto_gptq.utils import import_utils  # noqa: E402


@custom_patch(
    name="auto_gptq.utils.import_utils.dynamically_import_QuantLinear",
    reason=(),
    required=True,
    env_var_to_disable="DISABLE_AUTO_GPTQ_QUANTLINEAR_IMPORT_PATCH",
)
def patch_dynamically_import_quantlinear() -> None:
    def patched_dynamically_import_quantlinear(
        use_triton: bool,
        desc_act: bool,
        group_size: int,
        bits: int,
        disable_exllama: bool | None = None,
        disable_exllamav2: bool = False,
        use_qigen: bool = False,
        disable_marlin: bool = True,
    ):
        if not desc_act or group_size == -1:
            from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
        else:
            from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
        return QuantLinear

    import_utils.dynamically_import_QuantLinear = patched_dynamically_import_quantlinear
