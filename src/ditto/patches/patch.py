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
