import os
import pytest

from agents.anthropic_client import AnthropicClient
from agents.general_model_client import Message, GenerationConfig


# Common model name to use in all integration tests.
# You can change this to another Claude 3 model if needed.
MODEL_NAME = "claude-3-haiku-20240307"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _create_real_client() -> AnthropicClient:
    """
    Create a real AnthropicClient using the API key from environment.

    This is used across all integration tests to ensure we call the
    actual Anthropic API, not a mock.
    """
    api_key = os.environ["ANTHROPIC_API_KEY"]
    return AnthropicClient(
        api_key=api_key,
        default_model=MODEL_NAME,
    )


skip_if_no_key = pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ
    or not os.environ["ANTHROPIC_API_KEY"]
    or os.environ["ANTHROPIC_API_KEY"] == "api_key",
    reason=(
        "Real API key not provided via environment variable ANTHROPIC_API_KEY "
        "or still set to placeholder 'api_key' from launch.json"
    ),
)


# -------------------------------------------------------------------------
# Integration: generate_text()
# -------------------------------------------------------------------------

@pytest.mark.integration
@skip_if_no_key
def test_real_generate_text_basic():
    """
    Integration test:
    Call AnthropicClient.generate_text() against the real Anthropic API.

    Verifies that:
    - A non-empty string is returned.
    - Usage information contains non-zero token counts.
    - Basic metadata is present.
    """

    client = _create_real_client()

    cfg = GenerationConfig(
        model=MODEL_NAME,
        max_tokens=32,
        temperature=0.2,
    )

    resp = client.generate_text("Say a short one-word greeting.", cfg)

    # Assertions similar to unit test, but now with the real API
    assert isinstance(resp.text, str)
    assert len(resp.text.strip()) > 0

    assert resp.usage is not None
    assert resp.usage.total_tokens > 0
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0

    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("model") == MODEL_NAME


# -------------------------------------------------------------------------
# Integration: chat(streaming=False)
# -------------------------------------------------------------------------

@pytest.mark.integration
@skip_if_no_key
def test_real_chat_non_streaming():
    """
    Integration test:
    Call AnthropicClient.chat() with streaming=False against the real API.

    Verifies that:
    - A ModelResponse is returned.
    - Response text is non-empty.
    - Usage and metadata are populated.
    """

    client = _create_real_client()

    messages = [
        Message(role="system", content="You are a concise assistant."),
        Message(role="user", content="Reply with a two-word answer."),
    ]
    cfg = GenerationConfig(
        model=MODEL_NAME,
        max_tokens=32,
        temperature=0.2,
    )

    resp = client.chat(messages, generation_config=cfg, streaming=False)

    assert isinstance(resp.text, str)
    assert len(resp.text.strip()) > 0

    assert resp.usage is not None
    assert resp.usage.total_tokens > 0
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0

    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("model") == MODEL_NAME


# -------------------------------------------------------------------------
# Integration: chat(streaming=True)
# -------------------------------------------------------------------------

@pytest.mark.integration
@skip_if_no_key
def test_real_chat_streaming():
    """
    Integration test:
    Call AnthropicClient.chat() with streaming=True against the real API.

    Verifies that:
    - Streaming yields at least one text_delta chunk.
    - Final event has type="done".
    """

    client = _create_real_client()

    messages = [
        Message(role="system", content="You are a concise assistant."),
        Message(role="user", content="Reply with a short sentence."),
    ]
    cfg = GenerationConfig(
        model=MODEL_NAME,
        max_tokens=64,
        temperature=0.2,
    )

    stream = client.chat(messages, generation_config=cfg, streaming=True)

    events = list(stream)

    # There should be at least one text_delta chunk and a final "done"
    assert len(events) >= 2

    text_chunks = [e.text_delta for e in events if e.type == "text_delta"]
    done_events = [e for e in events if e.type == "done"]

    assert len(text_chunks) > 0
    assert all(isinstance(chunk, str) and chunk for chunk in text_chunks)
    assert len(done_events) == 1
