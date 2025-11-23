from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, Awaitable, Iterable


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

@dataclass
class StreamEvent:
    """
    Single streaming event from the model.

    Semantics:
    - type="text_delta": `text_delta` contains the next piece of text.
    - type="done": streaming is finished.
    - type="error": an error occurred; details in metadata/raw.
    """

    type: Literal["text_delta", "done", "error"]
    text_delta: str | None = None
    raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class ModelClient(ABC):
    """
    Abstract interface for LLM / multimodal providers.

    Concrete implementations (AnthropicClient, OpenAIClient, GeminiClient, etc.)
    must implement the abstract methods below.
    """
    def __init__(self, default_model: str | None = None, **client_options: Any) -> None:
        """
        `default_model`:
            Model name used when GenerationConfig.model is not provided.

        `client_options`:
            Provider-specific options such as api_key, base_url, timeouts, etc.
        """
        self.default_model = default_model
        self._client_options = client_options

    @abstractmethod
    def generate_text(self, prompt: str, config: GenerationConfig) -> ModelResponse:
        """Synchronous text generation."""
    
    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        generation_config: GenerationConfig | None = None,
        streaming: bool = True,
    ) -> ModelResponse | Iterable[StreamEvent]:
        """
        Chat-based text generation.

        Semantics:
        - If `streaming` is False:
            Return a fully assembled ModelResponse with the complete text.
        - If `streaming` is True:
            Return an Iterable[StreamEvent] that yields incremental events
            (e.g. text_delta chunks) as the model generates output.

        Implementations are free to:
        - Ignore streaming=True and still return a full ModelResponse, OR
        - Properly implement streaming and return an iterator / generator
          of StreamEvent objects.
        """
    
    def embed(
        self,
        texts: Sequence[str],
        generation_config: GenerationConfig | None = None,
    ) -> EmbeddingResponse:
        """Compute text embeddings for a batch of input strings."""
    
    @abstractmethod
    def embed_image(
        self,
        images: Sequence[Any],
        generation_config: GenerationConfig | None = None,
    ) -> EmbeddingResponse:
    
        """Compute image embeddings for a batch of input images."""
    
    # --- Image Operations ---

    @abstractmethod
    def upload_image(
        self,
        content_or_path: str | bytes,
        purpose: str,
        options: dict[str, Any] | None = None,
    ) -> FileRef:
        """Upload an image to the provider's storage and return a FileRef."""

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> ImageResponse:
        """Generate one or more images from a text prompt."""

    @abstractmethod
    def edit_image(
        self,
        image: FileRef | bytes | str,
        instructions: str,
        generation_config: GenerationConfig | None = None,
    ) -> ImageResponse:
        """Edit an existing image according to the given instructions."""

    @abstractmethod
    def image_variations(
        self,
        image: FileRef | bytes | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[ImageResponse]:
        """Generate multiple variations of the same input image."""

    @abstractmethod
    def analyze_image(
        self,
        image_or_file_id: FileRef | str | bytes,
        generation_config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """Analyze an image and return a textual description / analysis."""

    # --- File Operations ---

    @abstractmethod
    def upload_file(
        self,
        content_or_path: str | bytes,
        purpose: str,
        generation_config: GenerationConfig | None = None,
    ) -> FileRef:
        """Upload a generic file (PDF, text, etc.) and return a FileRef."""

    @abstractmethod
    def get_file(self, file_id: str) -> FileRef:
        """Fetch metadata for a stored file."""

    @abstractmethod
    def download_file(self, file_id: str) -> bytes | str:
        """
        Download file contents.

        The concrete implementation decides whether to return bytes or str,
        depending on file type and use-case.
        """

    @abstractmethod
    def list_files(self, filters: dict[str, Any] | None = None) -> list[FileRef]:
        """List available files, optionally filtered by provider-specific criteria."""

    @abstractmethod
    def delete_file(self, file_id: str) -> bool:
        """Delete a file. Return True on success."""

    @abstractmethod
    def update_file_metadata(
        self,
        file_id: str,
        metadata: dict[str, Any],
    ) -> FileRef:
        """Update file metadata such as name, description, or tags."""

    @abstractmethod
    def generate_from_file(
        self,
        file_id: str,
        generation_config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """
        Run the model on a file (e.g. summarize a PDF, extract insights, etc.).
        """

    # --- Helper for multimodal chat ---

    def attach_file_to_message(self, message: Message, file_ref: FileRef) -> Message:
        """
        Helper: attach a FileRef to a Message via its metadata.

        Concrete implementations can override this if they need
        a different attachment schema.
        """
        attachments = message.metadata.setdefault("attachments", [])
        attachments.append(
            {
                "file_id": file_ref.id,
                "filename": file_ref.filename,
                "purpose": file_ref.purpose,
            }
        )
        return message

    # --- Billing / Cost Estimation ---

    @abstractmethod
    def estimate_cost(self, request: dict[str, Any]) -> CostInfo:
        """
        Estimate the cost of a request.

        `request` is an abstract description (e.g. token counts, number of images),
        which concrete implementations interpret according to provider pricing.
        """

    # --- Execution Modes ---

    def execute_sync(self, request: dict[str, Any]) -> Any:
        """
        Generic sync execution entry point.

        This provides an optional "router-style" entry that can be convenient
        for agents or higher-level orchestrators that construct a generic
        request dict and don't want to call specific methods directly.
        """
        kind = request.get("kind")
        cfg = request.get("generation_config")

        if kind == "generate_text":
            return self.generate_text(request["prompt"], cfg)
        if kind == "chat":
            return self.chat(
                request["messages"],
                cfg,
                streaming=request.get("streaming", False),
            )
        if kind == "embed":
            return self.embed(request["texts"], cfg)
        if kind == "generate_image":
            return self.generate_image(request["prompt"], cfg)

        raise ValueError(f"Unsupported request kind: {kind!r}")

    @abstractmethod
    def execute_async(self, request: dict[str, Any]) -> Awaitable[Any]:
        """
        Generic async execution entry point.

        Implementations can mirror the logic of `execute_sync` but use
        async SDKs / HTTP clients under the hood.
        """

    # --- Model Information ---

    @abstractmethod
    def get_model(self, model_name: str) -> ModelInfo:
        """Return metadata about a specific model."""