from __future__ import annotations

import os
import asyncio
from typing import Any

import anthropic

from .general_model_client import (
    ModelClient,
    Message,
    GenerationConfig,
    ModelResponse,
    UsageInfo,
    EmbeddingResponse,
    ImageResponse,
    FileRef,
    CostInfo,
    ModelInfo,
)
from collections.abc import Sequence

class AnthropicClient(ModelClient):
    """Client for interacting with Anthropic's Claude models."""
    def __init__(
            self,
            api_key: str,
            default_model: str,
            **client_options: Any,
        ) -> None:
            """
            `api_key`:
                Anthropic API key. If not provided, ANTHROPIC_API_KEY env var is used.

            `default_model`:
                Default model name used when GenerationConfig.model is not set.

            `client_options`:
                Extra options passed to the underlying Anthropic client
                (e.g. base_url, timeout, default_headers, etc.).
            """
            super().__init__(default_model=default_model, **client_options)
            self._anthropic = anthropic.Anthropic(api_key=api_key, **client_options)
        
    def _resolve_max_tokens(self, config: GenerationConfig | None) -> int:
        """Resolve max_tokens with a safe default."""
        if config and config.max_tokens is not None:
            return config.max_tokens
        # Safe default; caller can always override in GenerationConfig
        return 1024
    
    