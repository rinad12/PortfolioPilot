import os
import pytest

from agents.gemini_client import GeminiClient
from agents.general_model_client import Message, GenerationConfig

from agents.general_model_client import (
    Message,
    GenerationConfig,
    ModelResponse,
    EmbeddingResponse,
    FileRef,
    StreamEvent,
)

@pytest.fixture
def mock_genai_client(monkeypatch):
    """
    Patch google.genai.Client inside the gemini_client module and return a
    single shared fake client instance.

    This prevents real network calls and lets us assert how GeminiClient
    uses the SDK.
    """



    class FakeUsage:
        def __init__(self, prompt=0, completion=0, total=0):
            self.prompt_token_count = prompt
            self.candidates_token_count = completion
            self.total_token_count = total

    class FakeModels:
        def __init__(self):
            self.generate_content = MagicMock()
            self.generate_content_stream = MagicMock()
            self.embed_content = MagicMock()
            self.generate_images = MagicMock()
            self.get = MagicMock()

    class FakeFiles:
        def __init__(self):
            self.upload = MagicMock()
            self.get = MagicMock()
            self.list = MagicMock()
            self.delete = MagicMock()

    class FakeAioModels:
        async def generate_content(self, *args, **kwargs):
            class Resp:
                text = "async response"
                usage_metadata = FakeUsage(prompt=1, completion=2, total=3)

            return Resp()

    class FakeAio:
        def __init__(self):
            self.models = FakeAioModels()

    class FakeClient:
        def __init__(self, **kwargs):
            # kwargs will contain api_key, etc. We simply ignore them in tests.
            self.models = FakeModels()
            self.files = FakeFiles()
            self.aio = FakeAio()

    fake_client = FakeClient()

    # We want to inspect constructor call as well, so wrap it in MagicMock
    client_ctor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(GeminiClient.genai, "Client", client_ctor)

    # Expose helpers to tests
    fake_client.FakeUsage = FakeUsage
    fake_client._client_ctor = client_ctor
    return fake_client


def test_generate_text_basic(mock_genai_client):
    """
    Ensure generate_text:
      - calls underlying models.generate_content with expected arguments
      - returns a ModelResponse with correct text and usage mapping.
    """

    class FakeResp:
        def __init__(self, text, usage):
            self.text = text
            self.usage_metadata = usage

    usage = mock_genai_client.FakeUsage(prompt=5, completion=7, total=12)
    fake_resp = FakeResp("hello from gemini", usage)
    mock_genai_client.models.generate_content.return_value = fake_resp

    client = GeminiClient(default_model="gemini-2.0-flash", api_key="dummy-key")
    config = GenerationConfig(temperature=0.3, max_tokens=64)

    # Act
    result = client.generate_text("Some prompt", config)

    # Assert: return type and content
    assert isinstance(result, ModelResponse)
    assert result.text == "hello from gemini"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 12

    # Assert: underlying SDK was called correctly
    mock_genai_client.models.generate_content.assert_called_once()
    call_args = mock_genai_client.models.generate_content.call_args
    assert call_args.kwargs["model"] == "gemini-2.0-flash"
    assert call_args.kwargs["contents"] == "Some prompt"
    assert "config" in call_args.kwargs


def test_chat_non_streaming(mock_genai_client):
    """
    Ensure chat(..., streaming=False) returns a full ModelResponse
    and correctly transforms Message roles into Gemini contents.
    """

    class FakeResp:
        def __init__(self, text, usage):
            self.text = text
            self.usage_metadata = usage

    usage = mock_genai_client.FakeUsage(prompt=10, completion=15, total=25)
    fake_resp = FakeResp("chat reply", usage)
    mock_genai_client.models.generate_content.return_value = fake_resp

    client = GeminiClient(default_model="gemini-2.0-flash", api_key="dummy-key")
    cfg = GenerationConfig(temperature=0.5)

    messages = [
        Message(role="system", content="You are a helpful bot."),
        Message(role="user", content="Hello!"),
        Message(role="assistant", content="Hi, user."),
    ]

    # Act
    result = client.chat(messages, generation_config=cfg, streaming=False)

    # Assert: we get a full response
    assert isinstance(result, ModelResponse)
    assert result.text == "chat reply"
    assert result.usage.total_tokens == 25

    # Assert: SDK got contents and config
    mock_genai_client.models.generate_content.assert_called_once()
    call_args = mock_genai_client.models.generate_content.call_args
    kwargs = call_args.kwargs

    assert kwargs["model"] == "gemini-2.0-flash"
    contents = kwargs["contents"]
    # system message is mapped into system_instruction, so contents should only contain non-system messages
    assert len(contents) == 2  # user + assistant

    first_content = contents[0]
    assert first_content.role == "user"
    assert first_content.parts[0].text == "Hello!"

    second_content = contents[1]
    assert second_content.role == "model"
    assert second_content.parts[0].text == "Hi, user."

    # config must contain system_instruction merged from system messages
    config = kwargs["config"]
    assert hasattr(config, "system_instruction")
    assert "You are a helpful bot." in config.system_instruction


def test_chat_streaming(mock_genai_client):
    """
    Ensure chat(..., streaming=True) returns an iterator over StreamEvent
    and produces a final 'done' event.
    """

    class FakeChunk:
        def __init__(self, text):
            self.text = text
            self.usage_metadata = None

    stream_chunks = [FakeChunk("Hello "), FakeChunk("world!")]
    mock_genai_client.models.generate_content_stream.return_value = stream_chunks

    client = GeminiClient(default_model="gemini-2.0-flash", api_key="dummy-key")

    messages = [Message(role="user", content="Say hello")]

    # Act
    events = list(client.chat(messages, streaming=True))

    # Assert: we expect 2 text_delta events + 1 done
    assert len(events) == 3
    assert isinstance(events[0], StreamEvent)
    assert events[0].type == "text_delta"
    assert events[0].text_delta == "Hello "

    assert events[1].type == "text_delta"
    assert events[1].text_delta == "world!"

    assert events[2].type == "done"


def test_embed_text(mock_genai_client):
    """
    Ensure embed() calls models.embed_content and returns EmbeddingResponse
    with the correct nested list of floats.
    """

    class FakeEmbedding:
        def __init__(self, values):
            self.values = values

    class FakeResp:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    fake_emb1 = FakeEmbedding([0.1, 0.2, 0.3])
    fake_emb2 = FakeEmbedding([0.4, 0.5, 0.6])
    mock_genai_client.models.embed_content.return_value = FakeResp(
        [fake_emb1, fake_emb2]
    )

    client = GeminiClient(
        default_model="gemini-2.0-flash",
        embedding_model="text-embedding-004",
        api_key="dummy-key",
    )

    resp = client.embed(["text-1", "text-2"])

    assert isinstance(resp, EmbeddingResponse)
    assert len(resp.embeddings) == 2
    assert resp.embeddings[0] == [0.1, 0.2, 0.3]
    assert resp.embeddings[1] == [0.4, 0.5, 0.6]

    mock_genai_client.models.embed_content.assert_called_once()
    kwargs = mock_genai_client.models.embed_content.call_args.kwargs
    assert kwargs["model"] == "text-embedding-004"
    assert kwargs["contents"] == ["text-1", "text-2"]


def test_upload_file(mock_genai_client):
    """
    Ensure upload_file() uses Files API and returns a FileRef with correct fields.
    """

    class FakeFile:
        def __init__(self, name, display_name, size_bytes):
            self.name = name
            self.display_name = display_name
            self.size_bytes = size_bytes

    fake_file = FakeFile("files/123", "myfile.pdf", 42)
    mock_genai_client.files.upload.return_value = fake_file

    client = GeminiClient(default_model="gemini-2.0-flash", api_key="dummy-key")

    ref = client.upload_file("path/to/myfile.pdf", purpose="test")

    assert isinstance(ref, FileRef)
    assert ref.id == "files/123"
    assert ref.filename == "myfile.pdf"
    assert ref.bytes_size == 42
    assert ref.purpose == "test"

    mock_genai_client.files.upload.assert_called_once()
    kwargs = mock_genai_client.files.upload.call_args.kwargs
    assert kwargs["file"] == "path/to/myfile.pdf"


def test_estimate_cost_basic():
    """
    Ensure estimate_cost() returns a CostInfo with zero price and
    copies token information into breakdown.
    """

    client = GeminiClient(default_model="gemini-2.0-flash", api_key="dummy-key")

    request = {
        "model": "gemini-2.0-flash",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        },
    }

    cost = client.estimate_cost(request)

    assert cost.estimated_price == 0.0
    assert cost.currency == "USD"
    assert cost.breakdown["model"] == "gemini-2.0-flash"
    assert cost.breakdown["prompt_tokens"] == 100
    assert cost.breakdown["completion_tokens"] == 50
    assert "note" in cost.breakdown
