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

import os
from collections.abc import Callable
from typing import Any

from loguru import logger


def custom_patch(
    name: str,
    reason: str,
    required: bool,
    env_var_to_disable: str,
) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    disabled = os.getenv(env_var_to_disable, "0" if required else "1") == "1"

    def may_apply(patch_fn: Callable[[], Any]) -> Callable[[], Any]:
        if not disabled:
            patch_fn()
            logger.info(
                f"Applied custom patch for {name}. "
                f"To disable this patch, set the environment variable {env_var_to_disable}=1"
            )
        elif required:
            logger.warning(f"The custom patch for {name} is disabled, but it is required for {reason}")

        return patch_fn

    return may_apply
