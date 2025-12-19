from __future__ import annotations
from typing import Any, Sequence, Iterable, Awaitable
import asyncio

from google import genai
from google.genai import types as genai_types

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


class GeminiClient(ModelClient):
    """
    Concrete implementation of ModelClient for Google Gemini using the `google-genai` SDK.

    Supports:
      - text generation
      - chat
      - text embeddings
      - image generation (Imagen)
      - file upload / listing (limited by Gemini Files API)
      - async execution

    Limitations:
      - Image embeddings are not available in Gemini API
      - Files API does not support downloading file contents
      - Metadata updates are not supported

    """

    def __init__(
        self,
        default_model: str | None = "gemini-2.0-flash",
        embedding_model: str | None = "text-embedding-004",
        **client_options: Any,
    ) -> None:
        """
        Initialize the Gemini client.

        Parameters:
            default_model: default Gemini model for text/chat operations.
            embedding_model: default model for embedding operations.
            client_options: passed directly to genai.Client (e.g. api_key, vertexai=True, project, location).
        """
        super().__init__(default_model=default_model, **client_options)
        self._client = genai.Client(**client_options)
        self._embedding_model = embedding_model or "text-embedding-004"

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _resolve_model(self, config: GenerationConfig | None) -> str:
        """
        Determine which model should be used based on GenerationConfig or default_model.
        """
        model = (config.model if config and config.model else self.default_model)
        if not model:
            raise ValueError("GeminiClient: no model specified.")
        return model

    def _build_generate_config(
        self, config: GenerationConfig | None
    ) -> genai_types.GenerateContentConfig | None:
        """
        Convert the unified GenerationConfig into Google's GenerateContentConfig.
        """
        if config is None:
            return None

        extra = dict(config.extra or {})

        # Map temperature, top_p, max tokens
        if config.temperature is not None:
            extra.setdefault("temperature", config.temperature)
        if config.max_tokens is not None:
            extra.setdefault("max_output_tokens", config.max_tokens)
        if config.top_p is not None:
            extra.setdefault("top_p", config.top_p)

        return genai_types.GenerateContentConfig(**extra)

    def _map_usage(self, usage: Any | None) -> UsageInfo | None:
        """
        Convert Gemini's usage metadata into a unified UsageInfo instance.
        """
        if usage is None:
            return None

        return UsageInfo(
            prompt_tokens=getattr(usage, "prompt_token_count", 0),
            completion_tokens=getattr(usage, "candidates_token_count", 0),
            total_tokens=getattr(usage, "total_token_count", 0),
            extra={"raw": usage},
        )

    # ----------------------------------------------------------------------
    # Text generation
    # ----------------------------------------------------------------------

    def generate_text(self, prompt: str, config: GenerationConfig) -> ModelResponse:
        """
        Synchronous text generation wrapper using models.generate_content().
        """
        model = self._resolve_model(config)
        gen_config = self._build_generate_config(config)

        resp = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=gen_config,
        )

        return ModelResponse(
            text=resp.text,
            raw=resp,
            usage=self._map_usage(getattr(resp, "usage_metadata", None)),
        )

    # ----------------------------------------------------------------------
    # Chat
    # ----------------------------------------------------------------------

    def _messages_to_contents(
        self, messages: Sequence[Message]
    ) -> tuple[list[genai_types.Content], str | None]:
        """
        Convert a sequence of Message objects into Gemini's Content[] format.
        Also merges all system-role messages into a single system_instruction.
        """
        system_messages = []
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg.content)
                continue

            # Gemini expects "model" instead of "assistant"
            role = (
                "model" if msg.role == "assistant"
                else "tool" if msg.role == "tool"
                else "user"
            )

            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=msg.content)],
                )
            )

        system_instruction = "\n\n".join(system_messages) if system_messages else None
        return contents, system_instruction

    def chat(
        self,
        messages: Sequence[Message],
        generation_config: GenerationConfig | None = None,
        streaming: bool = True,
    ) -> ModelResponse | Iterable[StreamEvent]:
        """
        Chat completion wrapper.
        Supports both streaming and non-streaming modes.
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_model(cfg)

        contents, system_instruction = self._messages_to_contents(messages)
        gen_config = self._build_generate_config(cfg)

        # Insert the system instruction if present
        if system_instruction:
            if gen_config is None:
                gen_config = genai_types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            else:
                gen_config.system_instruction = system_instruction

        # ---- Streaming mode ----
        if streaming:
            def _stream():
                """
                Generator that yields StreamEvent instances as they arrive.
                """
                try:
                    for chunk in self._client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=gen_config,
                    ):
                        if getattr(chunk, "text", None):
                            yield StreamEvent(
                                type="text_delta",
                                text_delta=chunk.text,
                                raw=chunk,
                            )
                    yield StreamEvent(type="done")
                except Exception as exc:
                    yield StreamEvent(
                        type="error",
                        raw=exc,
                        metadata={"error": str(exc)},
                    )

            return _stream()

        # ---- Non-streaming mode ----
        resp = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config,
        )
        return ModelResponse(
            text=resp.text,
            raw=resp,
            usage=self._map_usage(getattr(resp, "usage_metadata", None)),
        )

    # ----------------------------------------------------------------------
    # Text embeddings
    # ----------------------------------------------------------------------

    def embed(
        self,
        texts: Sequence[str],
        generation_config: GenerationConfig | None = None,
    ) -> EmbeddingResponse:
        """
        Text embedding wrapper using models.embed_content().
        """
        model = self._embedding_model
        if generation_config and generation_config.model:
            model = generation_config.model

        resp = self._client.models.embed_content(
            model=model,
            contents=list(texts),
        )

        # Extract embedding vectors
        vectors = []
        for emb in getattr(resp, "embeddings", []):
            vectors.append(list(emb.values))

        return EmbeddingResponse(
            embeddings=vectors,
            raw=resp,
            usage=None,
        )

    # ----------------------------------------------------------------------
    # Image embeddings (not supported)
    # ----------------------------------------------------------------------

    def embed_image(
        self,
        images: Sequence[Any],
        generation_config: GenerationConfig | None = None,
    ) -> EmbeddingResponse:
        """
        Gemini does not currently provide generic image-embedding models.
        """
        raise NotImplementedError(
            "GeminiClient.embed_image: image embeddings are not supported in Gemini API."
        )

    # ----------------------------------------------------------------------
    # Image generation (Imagen)
    # ----------------------------------------------------------------------

    def upload_image(
        self,
        content_or_path: str | bytes,
        purpose: str,
        options: dict[str, Any] | None = None,
    ) -> FileRef:
        """
        Upload an image using the Files API.

        Files cannot be downloaded later; they exist only as
        stored references for model inputs.
        """
        file_obj = self._client.files.upload(file=content_or_path)

        return FileRef(
            id=file_obj.name,
            filename=getattr(file_obj, "display_name", file_obj.name),
            bytes_size=getattr(file_obj, "size_bytes", None),
            purpose=purpose,
            metadata={"raw": file_obj},
        )

    def generate_image(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> ImageResponse:
        """
        Generate images using an Imagen model (e.g., 'imagen-3.0-generate-001').
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_model(cfg)

        extra = dict(cfg.extra or {})
        img_config = genai_types.GenerateImagesConfig(**extra) if extra else None

        resp = self._client.models.generate_images(
            model=model,
            prompt=prompt,
            config=img_config,
        )

        images = [gi.image for gi in getattr(resp, "generated_images", [])]

        return ImageResponse(
            images=images,
            raw=resp,
        )

    def edit_image(self, image, instructions, generation_config=None) -> ImageResponse:
        """
        Image editing currently requires specialized Imagen models not
        exposed in the general SDK.
        """
        raise NotImplementedError("GeminiClient.edit_image: not implemented.")

    def image_variations(self, image, generation_config=None) -> list[ImageResponse]:
        """
        Image variations require dedicated Imagen APIs.
        """
        raise NotImplementedError("GeminiClient.image_variations: not implemented.")

    def analyze_image(
        self,
        image_or_file_id: FileRef | str | bytes,
        generation_config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """
        Vision-style prompt: send an image and ask the model to describe it.
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_model(cfg)
        gen_config = self._build_generate_config(cfg)

        # Use file raw object if provided
        if isinstance(image_or_file_id, FileRef):
            raw_file = image_or_file_id.metadata.get("raw")
            image_part = raw_file if raw_file is not None else image_or_file_id.id
        else:
            image_part = image_or_file_id

        resp = self._client.models.generate_content(
            model=model,
            contents=[
                "Describe this image in detail.",
                image_part,
            ],
            config=gen_config,
        )

        return ModelResponse(
            text=resp.text,
            raw=resp,
            usage=self._map_usage(getattr(resp, "usage_metadata", None)),
        )

    # ----------------------------------------------------------------------
    # File operations
    # ----------------------------------------------------------------------

    def upload_file(self, content_or_path, purpose, generation_config=None) -> FileRef:
        """
        Upload any file via Gemini Files API.
        """
        file_obj = self._client.files.upload(file=content_or_path)

        return FileRef(
            id=file_obj.name,
            filename=getattr(file_obj, "display_name", file_obj.name),
            bytes_size=getattr(file_obj, "size_bytes", None),
            purpose=purpose,
            metadata={"raw": file_obj},
        )

    def get_file(self, file_id: str) -> FileRef:
        """
        Retrieve metadata about a stored file.
        """
        file_obj = self._client.files.get(name=file_id)

        return FileRef(
            id=file_obj.name,
            filename=getattr(file_obj, "display_name", file_obj.name),
            bytes_size=getattr(file_obj, "size_bytes", None),
            purpose=None,
            metadata={"raw": file_obj},
        )

    def download_file(self, file_id: str) -> bytes | str:
        """
        Files API does NOT allow downloading file contents.
        """
        raise NotImplementedError(
            "GeminiClient.download_file: downloading is not supported by Gemini Files API."
        )

    def list_files(self, filters=None) -> list[FileRef]:
        """
        List all uploaded files accessible to this client.
        """
        files = self._client.files.list()
        result = []

        for f in files:
            result.append(
                FileRef(
                    id=f.name,
                    filename=getattr(f, "display_name", f.name),
                    bytes_size=getattr(f, "size_bytes", None),
                    purpose=None,
                    metadata={"raw": f},
                )
            )

        return result

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file by ID.
        """
        self._client.files.delete(name=file_id)
        return True

    def update_file_metadata(self, file_id: str, metadata):
        """
        The Files API does not support metadata updates.
        """
        raise NotImplementedError("GeminiClient.update_file_metadata: not supported.")

    def generate_from_file(self, file_id: str, prompt: str, generation_config=None) -> ModelResponse:
        """
        Run the model on a file (typically summarization).
        """
        cfg = generation_config or GenerationConfig()
        model = self._resolve_model(cfg)
        gen_config = self._build_generate_config(cfg)

        
        file_obj = self._client.files.get(name=file_id)

        resp = self._client.models.generate_content(
            model=model,
            contents=[prompt, file_obj],
            config=gen_config,
        )

        return ModelResponse(
            text=resp.text,
            raw=resp,
            usage=self._map_usage(getattr(resp, "usage_metadata", None)),
        )

    # ----------------------------------------------------------------------
    # Cost estimation
    # ----------------------------------------------------------------------

    def estimate_cost(self, request: dict[str, Any]) -> CostInfo:
        raise NotImplementedError("GeminiClient.estimate_cost is not implemented.")

    # ----------------------------------------------------------------------
    # Async execution wrapper
    # ----------------------------------------------------------------------

    def execute_async(self, request: dict[str, Any]) -> Awaitable[Any]:
        """
        Async version of execute_sync using the asynchronous Gemini API.
        """

        kind = request.get("kind")
        cfg = request.get("generation_config")

        async def _run_generate_text():
            model = self._resolve_model(cfg)
            gen_config = self._build_generate_config(cfg)

            resp = await self._client.aio.models.generate_content(
                model=model,
                contents=request["prompt"],
                config=gen_config,
            )

            return ModelResponse(
                text=resp.text,
                raw=resp,
                usage=self._map_usage(getattr(resp, "usage_metadata", None)),
            )

        async def _run_chat():
            messages = request["messages"]
            model = self._resolve_model(cfg or GenerationConfig())
            contents, system_instruction = self._messages_to_contents(messages)

            gen_config = self._build_generate_config(cfg or GenerationConfig())
            if system_instruction:
                if gen_config is None:
                    gen_config = genai_types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                else:
                    gen_config.system_instruction = system_instruction

            resp = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )

            return ModelResponse(
                text=resp.text,
                raw=resp,
                usage=self._map_usage(getattr(resp, "usage_metadata", None)),
            )

        async def _run_embed():
            return self.embed(request["texts"], cfg)

        if kind == "generate_text":
            return _run_generate_text()
        if kind == "chat":
            return _run_chat()
        if kind == "embed":
            return _run_embed()

        async def _unsupported():
            raise ValueError(f"Unsupported async request kind: {kind!r}")

        return _unsupported()

    # ----------------------------------------------------------------------
    # Model information
    # ----------------------------------------------------------------------

    def get_model(self, model_name: str) -> ModelInfo:
        """
        Retrieve information about a Gemini model.
        """
        m = self._client.models.get(model=model_name)

        context_window = getattr(m, "input_token_limit", 0)
        supports_chat = "generateContent" in (getattr(m, "supported_actions", []) or [])

        return ModelInfo(
            name=model_name,
            provider="gemini",
            context_window=context_window,
            supports_chat=supports_chat,
            supports_embeddings=True,
            supports_images=True,
            supports_tools=True,
            metadata={"raw": m.model_dump() if hasattr(m, "model_dump") else m},
        )
