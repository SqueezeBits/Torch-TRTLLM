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

from typing import TypeVar

from loguru import logger
from transformers import PretrainedConfig

T = TypeVar("T")


def lookup_attributes(
    pretrained_config: PretrainedConfig | None,
    *names: str,
    default: T,
    not_found_ok: bool = False,
) -> T:
    """Look up attributes from a pretrained config, falling back to default if not found.

    Args:
        pretrained_config (PretrainedConfig | None): Config object to look up attributes from. Defaults to None.
        *names (str): Attribute names to search for
        default (T): Default value to return if attributes not found
        not_found_ok (bool): If True, suppress warning when attributes not found. Defaults to False.

    Returns:
        T: Found attribute value or default if not found
    """
    if pretrained_config is None:
        return default
    for name in names:
        if hasattr(pretrained_config, name):
            return getattr(pretrained_config, name)
    if not not_found_ok:
        logger.warning(
            "None of the following attributes are found in pretrained config. "
            f"Will use the default value {default}: {', '.join(names)}"
        )
    return default
