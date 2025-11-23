from __future__ import annotations

import os
import asyncio
from collections.abc import Sequence, Iterable
from typing import Any

import anthropic

from .model_client import (
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
    StreamEvent,
)

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
            self._client = anthropic.Anthropic(api_key=api_key, **client_options)

    def _resolve_model(self, cfg: GenerationConfig | None) -> str:
        """Pick model from GenerationConfig or fall back to default_model."""
        return cfg.model


        
    def _resolve_max_tokens(self, config: GenerationConfig | None) -> int:
        """Resolve max_tokens with a safe default."""
        if config and config.max_tokens is not None:
            return config.max_tokens
        # Safe default; caller can always override in GenerationConfig
        return 1024
    

    @staticmethod
    def _extract_text_from_message(message: Any) -> str:
        """
        Extract plain text from Anthropic Messages API response.

        Anthropic's response.content is a list of blocks:
        [{"type": "text", "text": "..."} , ...].
        We concatenate all text blocks.
        """
        parts: list[str] = []
        content = getattr(message, "content", None)
        if content is None:
            return ""

        for block in content:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type == "text":
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text", "")
                if text:
                    parts.append(text)
        return "".join(parts)
    
    @staticmethod
    def _build_usage(message: Any) -> UsageInfo:
        """
        Convert Anthropic usage to UsageInfo.

        Anthropic Messages API usage looks like:
        { "input_tokens": ..., "output_tokens": ... }.
        """
        usage = getattr(message, "usage", None) or {}
        input_tokens = getattr(usage, "input_tokens", None) or (
            usage.get("input_tokens") if isinstance(usage, dict) else 0
        )
        output_tokens = getattr(usage, "output_tokens", None) or (
            usage.get("output_tokens") if isinstance(usage, dict) else 0
        )
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)
        return UsageInfo(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    
    def _build_common_metadata(self, message: Any) -> dict[str, Any]:
        """Collect commonly useful metadata from Anthropic response."""
        meta: dict[str, Any] = {}
        for attr in ("id", "model", "stop_reason", "stop_sequence", "type", "role"):
            value = getattr(message, attr, None)
            if value is None and isinstance(message, dict):
                value = message.get(attr)
            if value is not None:
                meta[attr] = value
        return meta
    
    def _build_anthropic_messages(
        self,
        messages: Sequence[Message],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Convert our Message objects into Anthropic message list
        and a combined system prompt.
        """
        system_parts: list[str] = []
        anthro_messages: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role in ("user", "assistant"):
                anthro_messages.append(
                    {
                        "role": m.role,
                        "content": m.content,
                    }
                )
            # "tool" messages and other metadata are ignored for now

        system_prompt: str | None = "\n\n".join(system_parts) if system_parts else None
        return anthro_messages, system_prompt
    
    def generate_text(self, prompt: str, config: GenerationConfig | None = None) -> ModelResponse:
        """
        Single-shot text generation using Anthropic Messages API.

        Implementation detail:
        This is just a one-turn chat with a single user message.
        """
        model = self._resolve_model(config)
        max_tokens = self._resolve_max_tokens(config)

        extra = config.extra if config else {}
        temperature = config.temperature if config else 0.7
        top_p = config.top_p if config else 1.0

        msg = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **extra,
        )

        text = self._extract_text_from_message(msg)
        usage = self._build_usage(msg)
        metadata = self._build_common_metadata(msg)

        return ModelResponse(
            text=text,
            raw=msg,
            usage=usage,
            metadata=metadata,
        )
    
    def chat(
        self,
        messages: Sequence[Message],
        generation_config: GenerationConfig | None = None,
        streaming: bool = True,
    ) -> ModelResponse | Iterable[StreamEvent]:
        """
        Chat-based text generation.

        If `streaming` is False:
            - Performs a normal non-streaming Anthropic call.
            - Returns a single ModelResponse with the full output text.

        If `streaming` is True:
            - Returns an iterator of StreamEvent.
            - Each StreamEvent with type="text_delta" contains a piece of text.
            - Final event will have type="done".
        """
        model = self._resolve_model(generation_config)
        max_tokens = self._resolve_max_tokens(generation_config)

        extra = generation_config.extra if generation_config else {}
        temperature = generation_config.temperature if generation_config else 0.7
        top_p = generation_config.top_p if generation_config else 1.0

        anthro_messages, system_prompt = self._build_anthropic_messages(messages)

        if not streaming:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=anthro_messages,
                **extra,
            )
            if system_prompt:
                kwargs["system"] = system_prompt

            msg = self._client.messages.create(**kwargs)

            text = self._extract_text_from_message(msg)
            usage = self._build_usage(msg)
            metadata = self._build_common_metadata(msg)

            return ModelResponse(
                text=text,
                raw=msg,
                usage=usage,
                metadata=metadata,
            )
        
        # --- Streaming branch: return generator of StreamEvent ---

        def _stream() -> Iterable[StreamEvent]:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=anthro_messages,
                **extra,
            )
            if system_prompt:
                kwargs["system"] = system_prompt

            try:
                with self._client.messages.stream(**kwargs) as stream:
                    for event in stream:
                        # We care mainly about content_block_delta with text_delta
                        if event.type == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            delta_type = getattr(delta, "type", None) if delta is not None else None
                            if delta_type == "text_delta":
                                text_piece = getattr(delta, "text", "") or ""
                                if text_piece:
                                    yield StreamEvent(
                                        type="text_delta",
                                        text_delta=text_piece,
                                        raw=event,
                                    )
                    # When stream is exhausted, signal "done"
                    yield StreamEvent(type="done")
            except Exception as e:  # noqa: BLE001
                # On error, yield a single error event
                yield StreamEvent(
                    type="error",
                    text_delta=None,
                    raw=e,
                    metadata={"error": str(e)},
                )

        return _stream()
    
    def embed(
        self,
        texts: Sequence[str],
        generation_config: GenerationConfig | None = None,
    ) -> EmbeddingResponse:
        """
        Anthropic does not currently expose a native embeddings API.

        This method is intentionally left unimplemented. You can:
        - Integrate a separate embeddings provider (OpenAI, Voyage, etc),
        - Or compose another client inside AnthropicClient and delegate.
        """
        raise NotImplementedError("AnthropicClient.embed is not implemented (no native embeddings API).")

    def embed_image(
        self,
        images: Sequence[Any],
        generation_config: GenerationConfig | None = None,
    ) -> EmbeddingResponse:
        """
        Anthropic does not provide a dedicated image embedding API.

        Implement image embeddings via another provider if needed.
        """
        raise NotImplementedError("AnthropicClient.embed_image is not implemented.")

    # ---------------------------------------------------------------------
    # Image operations (not supported natively by Anthropic)
    # ---------------------------------------------------------------------

    def upload_image(
        self,
        content_or_path: str | bytes,
        purpose: str,
        options: dict[str, Any] | None = None,
    ) -> FileRef:
        raise NotImplementedError("AnthropicClient.upload_image is not implemented.")

    def generate_image(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> ImageResponse:
        raise NotImplementedError("AnthropicClient.generate_image is not implemented.")

    def edit_image(
        self,
        image: FileRef | bytes | str,
        instructions: str,
        generation_config: GenerationConfig | None = None,
    ) -> ImageResponse:
        raise NotImplementedError("AnthropicClient.edit_image is not implemented.")

    def image_variations(
        self,
        image: FileRef | bytes | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[ImageResponse]:
        raise NotImplementedError("AnthropicClient.image_variations is not implemented.")

    def analyze_image(
        self,
        image_or_file_id: FileRef | str | bytes,
        generation_config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """
        Claude can analyze images via the Messages API, but here we keep
        this unimplemented until a concrete image representation contract
        is decided (bytes vs URLs vs FileRef).
        """
        raise NotImplementedError("AnthropicClient.analyze_image is not implemented.")
    
    # ---------------------------------------------------------------------
    # File operations (Anthropic Files API – not wired yet)
    # ---------------------------------------------------------------------

    def upload_file(
        self,
        content_or_path: str | bytes,
        purpose: str,
        generation_config: GenerationConfig | None = None,
    ) -> FileRef:
        raise NotImplementedError("AnthropicClient.upload_file is not implemented.")

    def get_file(self, file_id: str) -> FileRef:
        raise NotImplementedError("AnthropicClient.get_file is not implemented.")

    def download_file(self, file_id: str) -> bytes | str:
        raise NotImplementedError("AnthropicClient.download_file is not implemented.")

    def list_files(self, filters: dict[str, Any] | None = None) -> list[FileRef]:
        raise NotImplementedError("AnthropicClient.list_files is not implemented.")

    def delete_file(self, file_id: str) -> bool:
        raise NotImplementedError("AnthropicClient.delete_file is not implemented.")

    def update_file_metadata(
        self,
        file_id: str,
        metadata: dict[str, Any],
    ) -> FileRef:
        raise NotImplementedError("AnthropicClient.update_file_metadata is not implemented.")

    def generate_from_file(
        self,
        file_id: str,
        generation_config: GenerationConfig | None = None,
    ) -> ModelResponse:
        raise NotImplementedError("AnthropicClient.generate_from_file is not implemented.")
    
    # ---------------------------------------------------------------------
    # Billing / cost estimation
    # ---------------------------------------------------------------------

    def estimate_cost(self, request: dict[str, Any]) -> CostInfo:
        """
        Placeholder for cost estimation.

        To implement:
        - Inspect request["kind"], model, token counts, etc.
        - Apply Anthropic's pricing table for the selected model.
        """
        raise NotImplementedError("AnthropicClient.estimate_cost is not implemented.")

    # ---------------------------------------------------------------------
    # Execution modes
    # ---------------------------------------------------------------------

    async def execute_async(self, request: dict[str, Any]) -> Any:
        """
        Async wrapper around execute_sync using a thread executor.

        For high-throughput applications, consider implementing a fully
        async client based on anthropic.AsyncAnthropic instead.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.execute_sync, request)

    # ---------------------------------------------------------------------
    # Model info
    # ---------------------------------------------------------------------

    def get_model(self, model_name: str) -> ModelInfo:
        """
        Minimal implementation returning basic metadata.

        You can extend this to:
        - Query a model registry or static config,
        - Differentiate context_window / capabilities per model.
        """
        return ModelInfo(
            name=model_name,
            provider="anthropic",
            context_window=200_000,  # rough default for Sonnet 3.5
            supports_chat=True,
            supports_embeddings=False,
            supports_images=True,   # Claude supports vision input
            supports_tools=True,    # tool calling is supported
            metadata={},
        )
    



    