from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, Awaitable


Role = Literal["user", "assistant", "system", "tool"]

# Dataclasses to hold agent configuration
@dataclass
class Message:
    """Single chat message in a conversation."""

    role: Role
    content: str
    # Arbitrary metadata, e.g. attachments, tool call info, message IDs, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationConfig:
    """
    Shared configuration for generation / embedding calls.

    Provider-specific options can be passed via `extra`.
    """

    model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class UsageInfo:
    """Token usage information returned by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelResponse:
    """Unified response for text / chat operations."""

    text: str
    # Raw provider response (for debugging or advanced use).
    raw: Any | None = None
    usage: UsageInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EmbeddingResponse:
    """Unified response for text/image embedding operations."""

    embeddings: list[list[float]]
    raw: Any | None = None
    usage: UsageInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ImageResponse:
    """
    Response for image-related operations.

    `images` may be URLs, bytes, or provider-specific handles,
    depending on how the concrete client is implemented.
    """

    images: list[Any]
    raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class FileRef:
    """Reference to a file stored by the provider."""

    id: str
    filename: str
    bytes_size: int | None = None
    purpose: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CostInfo:
    """Estimated cost for a request."""

    estimated_price: float
    currency: str = "USD"
    # Provider-specific breakdown, e.g. per-modality pricing.
    breakdown: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelInfo:
    """Metadata about a specific model available to the client."""

    name: str
    provider: str
    context_window: int
    supports_chat: bool
    supports_embeddings: bool
    supports_images: bool
    supports_tools: bool
    metadata: dict[str, Any] = field(default_factory=dict)

