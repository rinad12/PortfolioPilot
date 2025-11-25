from __future__ import annotations

from typing import Any, Iterable, Sequence, Awaitable

from openai import OpenAI, AsyncOpenAI

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
    StreamEvent,
)


class OpenAIClient(ModelClient):
    """
    Concrete implementation of ModelClient for OpenAI using the official `openai` SDK.

    Supported:
      - Text generation (generate_text)
      - Chat (chat, with optional streaming)
      - Text embeddings (embed)
      - Image generation / edit / variations
      - File upload / listing / download / delete

    Not supported or partially supported:
      - Image embeddings (embed_image -> NotImplementedError)
      - Cost estimation: basic stub implementation.
    """

    def __init__(
        self,
        default_model: str | None = "gpt-4.1-mini",
        embedding_model: str | None = "text-embedding-3-small",
        image_model: str | None = "gpt-image-1",
        **client_options: Any,
    ) -> None:
        """
        Parameters
        ----------
        default_model:
            Used for chat/text generation if GenerationConfig.model is None.

        embedding_model:
            Used for embedding calls if GenerationConfig.model is None.

        image_model:
            Used for image generation/edit/variations if not overridden in GenerationConfig.extra.

        client_options:
            Passed directly into `OpenAI(**client_options)` and `AsyncOpenAI(**client_options)`,
            e.g. api_key, base_url, organization, timeout, etc.
        """
        super().__init__(default_model=default_model, **client_options)
        self.embedding_model = embedding_model
        self.image_model = image_model

        # Sync and async OpenAI clients
        self._client = OpenAI(**client_options)
        self._async_client = AsyncOpenAI(**client_options)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _resolve_model(self, cfg: GenerationConfig | None) -> str:
        """Pick model from config or fall back to default_model."""
        if cfg is None:
            return self.default_model or "gpt-4.1-mini"
        return cfg.model or self.default_model or "gpt-4.1-mini"

    def _resolve_embedding_model(self, cfg: GenerationConfig | None) -> str:
        if cfg is None:
            return self.embedding_model or "text-embedding-3-small"
        return cfg.model or self.embedding_model or "text-embedding-3-small"

    def _resolve_image_model(self, cfg: GenerationConfig | None) -> str:
        if cfg is None:
            return self.image_model or "gpt-image-1"
        # Allow overriding via cfg.model or extra["image_model"]
        if cfg.model is not None:
            return cfg.model
        return cfg.extra.get("image_model", self.image_model or "gpt-image-1")

    @staticmethod
    def _map_usage(usage: Any | None) -> UsageInfo | None:
        """Convert OpenAI usage object into UsageInfo."""
        if usage is None:
            return None

        # OpenAI usage has fields like: prompt_tokens, completion_tokens, total_tokens
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

        # Keep the raw usage object in extra for debugging
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            extra={"raw": usage},
        )

    @staticmethod
    def _to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """
        Map our Message dataclass to OpenAI's chat messages format.

        NOTE: This is a minimal mapping (role + content).
        If you want tool calling, images, etc., you'll extend this.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            result.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    # You can use msg.metadata to build richer content in the future.
                }
            )
        return result

    # -------------------------------------------------------------------------
    # Text generation
    # -------------------------------------------------------------------------
    def generate_text(self, prompt: str, config: GenerationConfig) -> ModelResponse:
        """
        Simple text generation using chat.completions under the hood.

        We wrap the prompt into a single user message.
        """
        cfg = config or GenerationConfig()
        model = self._resolve_model(cfg)

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            **cfg.extra,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        usage = self._map_usage(response.usage)
        return ModelResponse(
            text=text,
            raw=response,
            usage=usage,
            metadata={"finish_reason": choice.finish_reason},
        )

    # -------------------------------------------------------------------------
    # Chat
    # -------------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        generation_config: GenerationConfig | None = None,
        streaming: bool = True,
    ) -> ModelResponse | Iterable[StreamEvent]:
        """
        Chat-based generation with optional streaming.

        If streaming=False, returns a single ModelResponse.
        If streaming=True, returns a generator of StreamEvent objects.
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_model(cfg)
        oa_messages = self._to_openai_messages(messages)

        if not streaming:
            response = self._client.chat.completions.create(
                model=model,
                messages=oa_messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                stream=False,
                **cfg.extra,
            )

            choice = response.choices[0]
            text = choice.message.content or ""
            usage = self._map_usage(response.usage)

            return ModelResponse(
                text=text,
                raw=response,
                usage=usage,
                metadata={"finish_reason": choice.finish_reason},
            )

        # --- Streaming mode ---
        def _event_iterator() -> Iterable[StreamEvent]:
            try:
                stream = self._client.chat.completions.create(
                    model=model,
                    messages=oa_messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    top_p=cfg.top_p,
                    stream=True,
                    **cfg.extra,
                )
                for chunk in stream:
                    # Each chunk has choices[0].delta.content with text diff
                    delta_obj = chunk.choices[0].delta
                    delta_text = delta_obj.content or ""
                    if delta_text:
                        yield StreamEvent(
                            type="text_delta",
                            text_delta=delta_text,
                            raw=chunk,
                        )
                yield StreamEvent(type="done")
            except Exception as exc:  # noqa: BLE001
                yield StreamEvent(
                    type="error",
                    text_delta=None,
                    raw=None,
                    metadata={"error": str(exc)},
                )

        return _event_iterator()

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        generation_config: GenerationConfig | None = None,
    ) -> EmbeddingResponse:
        """Text embeddings using OpenAI embeddings API."""
        model = self._resolve_embedding_model(generation_config)
        cfg = generation_config or GenerationConfig()

        response = self._client.embeddings.create(
            model=model,
            input=list(texts),
            **cfg.extra,
        )

        embeddings: list[list[float]] = [item.embedding for item in response.data]
        usage = self._map_usage(getattr(response, "usage", None))
        return EmbeddingResponse(
            embeddings=embeddings,
            raw=response,
            usage=usage,
        )

    def embed_image(
        self,
        images: Sequence[Any],
        generation_config: GenerationConfig | None = None,
    ) -> EmbeddingResponse:
        """
        OpenAI does not currently expose a public image-embedding endpoint.

        You can either:
          - Raise NotImplementedError as below, or
          - Implement a custom workaround (e.g. send images through a multimodal model
            and convert features yourself).
        """
        raise NotImplementedError("Image embeddings are not supported via OpenAI API yet.")

    # -------------------------------------------------------------------------
    # Image operations
    # -------------------------------------------------------------------------
    def upload_image(
        self,
        content_or_path: str | bytes,
        purpose: str,
        options: dict[str, Any] | None = None,
    ) -> FileRef:
        """
        For OpenAI, images are also uploaded through the generic files API.

        `purpose` is typically something like 'fine-tune' / 'assistants' / etc.
        """
        file_kwargs: dict[str, Any] = {"purpose": purpose}
        if isinstance(content_or_path, str):
            # Interpret as file path
            file_kwargs["file"] = open(content_or_path, "rb")
        else:
            # Raw in-memory bytes
            file_kwargs["file"] = content_or_path

        if options:
            file_kwargs.update(options)

        file_obj = self._client.files.create(**file_kwargs)

        return FileRef(
            id=file_obj.id,
            filename=file_obj.filename,
            bytes_size=file_obj.bytes,
            purpose=file_obj.purpose,
            metadata={"status": getattr(file_obj, "status", None)},
        )

    def generate_image(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> ImageResponse:
        """Text-to-image using `images.generate`."""
        cfg = generation_config or GenerationConfig()
        model = self._resolve_image_model(cfg)
        size = cfg.extra.get("size", "1024x1024")
        n = cfg.extra.get("n", 1)
        response_format = cfg.extra.get("response_format", "url")  # or "b64_json"

        response = self._client.images.generate(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
            **{k: v for k, v in cfg.extra.items() if k not in {"size", "n", "response_format"}},
        )

        # Collect URLs or base64 strings
        images: list[Any] = []
        for item in response.data:
            if response_format == "url":
                images.append(item.url)
            else:
                images.append(item.b64_json)

        return ImageResponse(images=images, raw=response)

    def edit_image(
        self,
        image: FileRef | bytes | str,
        instructions: str,
        generation_config: GenerationConfig | None = None,
    ) -> ImageResponse:
        """
        Image editing using `images.edits`.

        Note: OpenAI requires a PNG image and (optionally) a mask.
        For simplicity, this implementation assumes a single input image and no mask.
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_image_model(cfg)
        size = cfg.extra.get("size", "1024x1024")
        n = cfg.extra.get("n", 1)
        response_format = cfg.extra.get("response_format", "url")

        # Resolve image into file-like object
        img_file: Any
        if isinstance(image, FileRef):
            # Use Files API to download the bytes
            img_bytes = self.download_file(image.id)
            img_file = img_bytes
        elif isinstance(image, str):
            # Interpret as path
            img_file = open(image, "rb")
        else:
            img_file = image  # bytes

        response = self._client.images.edits(
            model=model,
            image=img_file,
            prompt=instructions,
            n=n,
            size=size,
            response_format=response_format,
        )

        images: list[Any] = []
        for item in response.data:
            images.append(item.url if response_format == "url" else item.b64_json)

        return ImageResponse(images=images, raw=response)

    def image_variations(
        self,
        image: FileRef | bytes | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[ImageResponse]:
        """
        Image variations using `images.variations`.
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_image_model(cfg)
        size = cfg.extra.get("size", "1024x1024")
        n = cfg.extra.get("n", 1)
        response_format = cfg.extra.get("response_format", "url")

        # Resolve image into file-like
        img_file: Any
        if isinstance(image, FileRef):
            img_bytes = self.download_file(image.id)
            img_file = img_bytes
        elif isinstance(image, str):
            img_file = open(image, "rb")
        else:
            img_file = image

        response = self._client.images.variations(
            model=model,
            image=img_file,
            n=n,
            size=size,
            response_format=response_format,
        )

        # For consistency, we return a list[ImageResponse],
        # each with a single image
        responses: list[ImageResponse] = []
        for item in response.data:
            single = item.url if response_format == "url" else item.b64_json
            responses.append(ImageResponse(images=[single], raw=response))

        return responses

    def analyze_image(
        self,
        image_or_file_id: FileRef | str | bytes,
        generation_config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """
        Analyze an image with a multimodal model.

        This is a minimal implementation that expects either:
          - A publicly accessible URL (str)
          - A FileRef whose id is a OpenAI file id of an image
        and sends it via chat as an image_url.

        For bytes / local paths, you'd need to upload them first.
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_model(cfg)

        # Normalise to URL or OpenAI file ID; here we only handle URL/file_id cases
        if isinstance(image_or_file_id, FileRef):
            image_url = {"file_id": image_or_file_id.id}
        elif isinstance(image_or_file_id, str):
            # Assume URL or file id; here we treat it as URL for simplicity
            image_url = {"url": image_or_file_id}
        else:
            raise NotImplementedError(
                "analyze_image currently expects a FileRef or URL string. "
                "Bytes/local paths are not supported in this minimal implementation."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        ]

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            **cfg.extra,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = self._map_usage(response.usage)

        return ModelResponse(
            text=text,
            raw=response,
            usage=usage,
            metadata={"finish_reason": choice.finish_reason},
        )

    # -------------------------------------------------------------------------
    # File operations
    # -------------------------------------------------------------------------
    def upload_file(
        self,
        content_or_path: str | bytes,
        purpose: str,
        generation_config: GenerationConfig | None = None,
    ) -> FileRef:
        """Upload a generic file to OpenAI Files API."""
        file_kwargs: dict[str, Any] = {"purpose": purpose}

        if isinstance(content_or_path, str):
            file_kwargs["file"] = open(content_or_path, "rb")
        else:
            file_kwargs["file"] = content_or_path

        response = self._client.files.create(**file_kwargs)

        return FileRef(
            id=response.id,
            filename=response.filename,
            bytes_size=response.bytes,
            purpose=response.purpose,
            metadata={"status": getattr(response, "status", None)},
        )

    def get_file(self, file_id: str) -> FileRef:
        """Fetch metadata for a stored file."""
        response = self._client.files.retrieve(file_id)

        return FileRef(
            id=response.id,
            filename=response.filename,
            bytes_size=response.bytes,
            purpose=response.purpose,
            metadata={"status": getattr(response, "status", None)},
        )

    def download_file(self, file_id: str) -> bytes | str:
        """Download file contents as bytes."""
        content = self._client.files.content(file_id)
        # `content` is an HTTPx Response-like object; we want the raw bytes
        return content.read()

    def list_files(self, filters: dict[str, Any] | None = None) -> list[FileRef]:
        """List available files."""
        response = self._client.files.list()
        refs: list[FileRef] = []
        for f in response.data:
            refs.append(
                FileRef(
                    id=f.id,
                    filename=f.filename,
                    bytes_size=f.bytes,
                    purpose=f.purpose,
                    metadata={"status": getattr(f, "status", None)},
                )
            )
        # You can implement client-side filtering based on `filters` if needed.
        return refs

    def delete_file(self, file_id: str) -> bool:
        """Delete a file. Returns True on success."""
        response = self._client.files.delete(file_id)
        return bool(getattr(response, "deleted", False))

    def update_file_metadata(
        self,
        file_id: str,
        metadata: dict[str, Any],
    ) -> FileRef:
        """
        OpenAI Files API has limited metadata updates.

        In the minimal implementation, we just fetch the file and
        attach the new metadata in the returned FileRef, but we do not
        persist it on OpenAI side (because there is no generic metadata update).
        """
        file_obj = self._client.files.retrieve(file_id)
        merged_meta = {"status": getattr(file_obj, "status", None)}
        merged_meta.update(metadata)

        return FileRef(
            id=file_obj.id,
            filename=file_obj.filename,
            bytes_size=file_obj.bytes,
            purpose=file_obj.purpose,
            metadata=merged_meta,
        )

    def generate_from_file(
        self,
        file_id: str,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """
        Very naive implementation: downloads the file and sends its content to the model.

        For large files, you should switch to the Assistants / Responses API instead.
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_model(cfg)

        # WARNING: this loads the full file into memory, which may not be safe for big files
        file_bytes = self.download_file(file_id)
        try:
            text_content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # As a fallback, just return a short message
            text_content = "[binary file content; cannot decode as UTF-8]"

        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            **cfg.extra,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = self._map_usage(response.usage)

        return ModelResponse(
            text=text,
            raw=response,
            usage=usage,
            metadata={"finish_reason": choice.finish_reason},
        )

    # -------------------------------------------------------------------------
    # Billing / cost estimation
    # -------------------------------------------------------------------------
    def estimate_cost(self, request: dict[str, Any]) -> CostInfo:
        raise NotImplementedError("OpenAIClient.estimate_cost is not implemented.")

    # -------------------------------------------------------------------------
    # Async execution
    # -------------------------------------------------------------------------
    async def execute_async(self, request: dict[str, Any]) -> Any:
        """
        Async version of execute_sync using AsyncOpenAI client.

        Supported kinds: generate_text, chat, embed, generate_image.
        """
        kind = request.get("kind")
        cfg: GenerationConfig | None = request.get("generation_config")

        if kind == "generate_text":
            prompt: str = request["prompt"]
            cfg = cfg or GenerationConfig()
            model = self._resolve_model(cfg)
            response = await self._async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                **cfg.extra,
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            usage = self._map_usage(response.usage)
            return ModelResponse(
                text=text,
                raw=response,
                usage=usage,
                metadata={"finish_reason": choice.finish_reason},
            )

        if kind == "chat":
            messages: Sequence[Message] = request["messages"]
            streaming: bool = request.get("streaming", False)
            cfg = cfg or GenerationConfig()
            model = self._resolve_model(cfg)
            oa_messages = self._to_openai_messages(messages)

            if streaming:
                # For simplicity, we do not implement async streaming here.
                # You can extend this to return an async generator of StreamEvent.
                raise NotImplementedError("Async streaming is not implemented yet.")

            response = await self._async_client.chat.completions.create(
                model=model,
                messages=oa_messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                **cfg.extra,
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            usage = self._map_usage(response.usage)
            return ModelResponse(
                text=text,
                raw=response,
                usage=usage,
                metadata={"finish_reason": choice.finish_reason},
            )

        if kind == "embed":
            texts: Sequence[str] = request["texts"]
            model = self._resolve_embedding_model(cfg)
            cfg = cfg or GenerationConfig()
            response = await self._async_client.embeddings.create(
                model=model,
                input=list(texts),
                **cfg.extra,
            )
            embeddings = [item.embedding for item in response.data]
            usage = self._map_usage(getattr(response, "usage", None))
            return EmbeddingResponse(
                embeddings=embeddings,
                raw=response,
                usage=usage,
            )

        if kind == "generate_image":
            prompt: str = request["prompt"]
            cfg = cfg or GenerationConfig()
            model = self._resolve_image_model(cfg)
            size = cfg.extra.get("size", "1024x1024")
            n = cfg.extra.get("n", 1)
            response_format = cfg.extra.get("response_format", "url")

            response = await self._async_client.images.generate(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                response_format=response_format,
                **{k: v for k, v in cfg.extra.items() if k not in {"size", "n", "response_format"}},
            )
            images: list[Any] = []
            for item in response.data:
                images.append(item.url if response_format == "url" else item.b64_json)
            return ImageResponse(images=images, raw=response)

        raise ValueError(f"Unsupported request kind for async execution: {kind!r}")

    # -------------------------------------------------------------------------
    # Model info
    # -------------------------------------------------------------------------
    def get_model(self, model_name: str) -> ModelInfo:
        """
        Fetch metadata about a specific model.

        This calls `client.models.retrieve` and maps a subset of fields.
        """
        model = self._client.models.retrieve(model_name)

        # Heuristics: OpenAI doesn't expose all these booleans directly,
        # so we approximate based on the model name / type.
        name = model.id
        provider = "openai"
        context_window = getattr(model, "context_length", 0) or 0

        lowered = name.lower()
        supports_embeddings = "embedding" in lowered
        supports_images = "image" in lowered
        supports_chat = not supports_embeddings and not supports_images
        supports_tools = supports_chat  # rough guess

        return ModelInfo(
            name=name,
            provider=provider,
            context_window=context_window,
            supports_chat=supports_chat,
            supports_embeddings=supports_embeddings,
            supports_images=supports_images,
            supports_tools=supports_tools,
            metadata={"raw": model},
        )
