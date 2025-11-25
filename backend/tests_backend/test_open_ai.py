import os
from collections.abc import Iterable

import pytest

from agents.openai_client import OpenAIClient
from agents.general_model_client import (
    GenerationConfig,
    Message,
    ModelResponse,
    EmbeddingResponse,
    StreamEvent,
    ModelInfo,
)


@pytest.fixture(scope="session")
def openai_client() -> OpenAIClient:
    """
    Integration fixture for OpenAIClient using a real API key.

    The key must be available as OPENAI_API_KEY in the environment.
    Tests will be skipped if the key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set in environment; skipping OpenAI integration tests")

    # You can override default_model / embedding_model if needed
    return OpenAIClient(
        api_key=api_key,
        # base_url can be overridden here if you proxy OpenAI
        # base_url=os.getenv("OPENAI_BASE_URL", None),
    )


@pytest.mark.integration
def test_generate_text_basic(openai_client: OpenAIClient) -> None:
    """
    Smoke test for generate_text: ensure we get a non-empty response and usage info.
    """
    cfg = GenerationConfig(
        # Explicit model is optional; you can rely on client's default_model.
        model="gpt-4.1-mini",
        max_tokens=16,
        temperature=0.0,
    )

    response = openai_client.generate_text("Say 'hello' in one short sentence.", cfg)

    assert isinstance(response, ModelResponse)
    assert isinstance(response.text, str)
    assert response.text.strip()  # should not be empty

    # Usage info is not guaranteed, but usually present
    if response.usage is not None:
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens >= response.usage.prompt_tokens


@pytest.mark.integration
def test_chat_non_streaming(openai_client: OpenAIClient) -> None:
    """
    Non-streaming chat should return a ModelResponse with text.
    """
    messages = [
        Message(role="user", content="Reply with the word 'ok' only."),
    ]
    cfg = GenerationConfig(
        model="gpt-4.1-mini",
        max_tokens=4,
        temperature=0.0,
    )

    result = openai_client.chat(messages, generation_config=cfg, streaming=False)

    assert isinstance(result, ModelResponse)
    text = result.text.strip().lower()
    assert "ok" in text


@pytest.mark.integration
def test_chat_streaming(openai_client: OpenAIClient) -> None:
    """
    Streaming chat should yield StreamEvent objects with text_deltas
    and end with a 'done' event.
    """
    messages = [
        Message(role="user", content="Answer with a very short greeting."),
    ]
    cfg = GenerationConfig(
        model="gpt-4.1-mini",
        max_tokens=16,
        temperature=0.0,
    )

    stream = openai_client.chat(messages, generation_config=cfg, streaming=True)

    assert isinstance(stream, Iterable)

    events = list(stream)
    assert events, "Streaming should yield at least one event"

    # Collect text deltas
    deltas = [e.text_delta for e in events if isinstance(e, StreamEvent) and e.text_delta]
    combined_text = "".join(deltas).strip()

    assert combined_text, "Combined streamed text should not be empty"

    # Last event should be 'done'
    assert isinstance(events[-1], StreamEvent)
    assert events[-1].type == "done"


@pytest.mark.integration
def test_embed_texts(openai_client: OpenAIClient) -> None:
    """
    Smoke test for embed: ensure we get the right number of embeddings
    and non-empty vectors.
    """
    texts = ["hello", "world"]
    cfg = GenerationConfig(
        # If model is None, OpenAIClient will use its embedding_model
        max_tokens=None,
    )

    response = openai_client.embed(texts, generation_config=cfg)

    assert isinstance(response, EmbeddingResponse)
    assert len(response.embeddings) == len(texts)
    first_vec = response.embeddings[0]
    assert isinstance(first_vec, list)
    assert len(first_vec) > 0  # embedding dimension should be > 0


@pytest.mark.integration
def test_get_model_info(openai_client: OpenAIClient) -> None:
    """
    Test get_model to ensure it returns a sensible ModelInfo object.
    """
    info = openai_client.get_model("gpt-4.1-mini")

    assert isinstance(info, ModelInfo)
    assert info.provider == "openai"
    assert isinstance(info.context_window, int)
    # At least one of these capabilities should be true for a chat model
    assert info.supports_chat is True
    assert info.supports_embeddings is False or isinstance(info.supports_embeddings, bool)
    assert info.supports_images is False or isinstance(info.supports_images, bool)


