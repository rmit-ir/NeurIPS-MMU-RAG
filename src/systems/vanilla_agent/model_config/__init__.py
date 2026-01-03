"""
Model configuration module for vanilla agent.

This module provides utilities to select the appropriate model configuration
(default_config or gpt_oss_config) based on model identifiers.
"""

from typing import Optional
from . import default_config, gpt_oss_config


def get_model_config(model_id: Optional[str] = None):
    """Get the appropriate model configuration based on model_id.

    Determines which config to use by checking if the model follows
    the GPT-OSS (OpenAI Harmony) format or standard OpenAI format.

    Args:
        model_id: Model identifier string (optional). If None or not a GPT-OSS model,
                 returns default_config.

    Returns:
        Either gpt_oss_config or default_config module based on model_id

    Examples:
        >>> config = get_model_config("gpt-oss")
        >>> # Returns gpt_oss_config

        >>> config = get_model_config("Qwen3-4B")
        >>> # Returns default_config
    """
    if model_id and gpt_oss_config.is_gpt_oss_model(model_id):
        return gpt_oss_config
    return default_config


__all__ = ['default_config', 'gpt_oss_config', 'get_model_config']
