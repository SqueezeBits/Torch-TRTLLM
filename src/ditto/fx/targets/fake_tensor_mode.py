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

import torch


def is_in_fake_tensor_mode() -> bool:
    """Check if the current dispatch mode is fake tensor mode.

    Returns:
        bool: True if the current dispatch mode is fake tensor mode, False otherwise.
    """
    # Get the current dispatch mode
    # Check if it's an instance of FakeTensorMode
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None
