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
import sys

os.environ["LOGURU_AUTOINIT"] = "0"
os.environ["LOGURU_LEVEL"] = os.getenv("DITTO_LOG_LEVEL", "INFO")

# pylint: disable-next=wrong-import-position
from loguru import logger  # noqa: E402

logger.add(
    sys.stdout,
    filter=lambda record: record["level"].name in ("INFO", "SUCCESS"),
    format="<m>ditto:{elapsed}</m> [{level}] <lvl>{message}</lvl>",
    colorize=True,
)

logger.add(
    sys.stdout,
    filter=lambda record: record["level"].name in ("TRACE", "DEBUG"),
    format="<m>ditto:{elapsed}</m> [{level}] [{name}/{function}:{line}] <lvl>{message}</lvl>",
    colorize=True,
)

logger.add(
    sys.stderr,
    filter=lambda record: record["level"].name in ("WARNING", "ERROR", "CRITICAL"),
    format="<m>ditto:{elapsed}</m> [{level}] [{name}/{function}:{line}] <lvl>{message}</lvl>",
    colorize=True,
)
