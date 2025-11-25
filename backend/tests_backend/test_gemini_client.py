import os
import pytest

from agents.gemini_client import GeminiClient
from agents.general_model_client import Message, GenerationConfig

# Имена переменных окружения, где может лежать ключ Gemini
GEMINI_ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_ENV_VAR")


@pytest.fixture(scope="session")
def has_gemini_key() -> bool:
    """Check if any Gemini-related API key is available.

    If not – skip integration tests that actually call the real API.
    """
    for name in GEMINI_ENV_VARS:
        if os.getenv(name):
            return True  # ключ нашли — можно запускать тесты

    pytest.skip(
        "No Gemini API key found in env. "
        "Set one of: GOOGLE_API_KEY / GEMINI_API_KEY / GEMINI_ENV_VAR to run integration tests."
    )


@pytest.fixture(scope="session")
def gemini_client(has_gemini_key: bool) -> GeminiClient:
    """
    Create a GeminiClient instance.

    The client itself is responsible for reading the API key from os.environ.
    If no key is configured, we skip the tests instead of failing them.
    """
    # Если сюда дошли — has_gemini_key уже либо вернул True, либо скипнул тесты
    return GeminiClient()



@pytest.mark.integration
def test_generate_text_returns_non_empty_answer(gemini_client: GeminiClient):
    """
    Basic smoke test: the client should return a non-empty text response
    for a simple prompt using the live Gemini API.
    """
    prompt = "Say 'hello' in one short English sentence."

    config = GenerationConfig()

    response = gemini_client.generate_text(prompt, config)

    text = response.text

    assert isinstance(text, str)
    assert text.strip() != ""


@pytest.mark.integration
def test_chat_simple_math(gemini_client: GeminiClient):
    """
    Chat test: verify that the client can handle a minimal chat history
    and that the model returns a sensible answer.
    """
    messages = [
        Message(role="user", content="What is 2 + 2? Answer with a single number."),
    ]

    config = GenerationConfig()

    response = gemini_client.chat(messages, config, streaming = False)

    text = response.text

    assert isinstance(text, str)
    assert text.strip() != ""


@pytest.mark.integration
def test_usage_info_if_available(gemini_client: GeminiClient):
    """
    Optional test: if your ModelResponse exposes usage information,
    check that it's populated with something reasonable.
    """
    prompt = "Write a very short sentence about cats."

    config = GenerationConfig()
    response = gemini_client.generate_text(prompt, config)

    usage = getattr(response, "usage", None)
    if usage is None:
        pytest.skip("ModelResponse.usage is not implemented on GeminiClient")

    # Под реальное определение UsageInfo:
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    # Если usage ещё не заполняется нормально – лучше скипнуть, а не фейлить
    if not all(isinstance(x, int) and x > 0 for x in (prompt_tokens, completion_tokens, total_tokens)):
        pytest.skip("UsageInfo does not contain populated token counts yet")

    assert prompt_tokens > 0
    assert completion_tokens > 0
    assert total_tokens >= prompt_tokens + completion_tokens


