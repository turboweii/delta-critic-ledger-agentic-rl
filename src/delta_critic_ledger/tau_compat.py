from __future__ import annotations

import inspect
import os
from typing import Any, Callable


def create_tau_env(get_env: Callable[..., Any], **kwargs: Any) -> Any:
    """Create a tau-bench environment across API versions."""
    parameters = inspect.signature(get_env).parameters
    user_api_base = kwargs.get("user_api_base")
    user_strategy = kwargs.get("user_strategy")
    if user_api_base and user_strategy != "human" and "user_api_base" not in parameters:
        os.environ["OPENAI_API_BASE"] = str(user_api_base)
        os.environ["OPENAI_BASE_URL"] = str(user_api_base)
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return get_env(**supported)
