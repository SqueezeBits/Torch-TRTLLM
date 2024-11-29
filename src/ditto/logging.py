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
    format="<m>ditto:{elapsed}</m> [{level}] [{name} - {function}:{line}] <lvl>{message}</lvl>",
    colorize=True,
)

logger.add(
    sys.stderr,
    filter=lambda record: record["level"].name in ("WARNING", "ERROR", "CRITICAL"),
    format="<m>ditto:{elapsed}</m> [{level}] [{name} - {function}:{line}] <lvl>{message}</lvl>",
    colorize=True,
)
