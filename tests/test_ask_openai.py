import pytest
from jipp.llms.openai_client import ask_openai, LLMError
from jipp.models.jipp_models import LLMMessage
from jipp.models.responses import LLMResponse


@pytest.mark.asyncio
async def test_ask_openai_basic():
    messages = [{"role": "user", "content": "Say 'Hello, World!'"}]
    response = await ask_openai(messages=messages, model="gpt-3.5-turbo")

    assert isinstance(response, LLMResponse)
    assert isinstance(response.message, LLMMessage)
    assert response.message.role == "assistant"
    assert "Hello, World!" in response.message.content
    assert response.usage is not None
    assert response.model == "gpt-3.5-turbo"
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_ask_openai_with_temperature():
    messages = [
        {"role": "user", "content": "Generate a random number between 1 and 10"}
    ]
    response = await ask_openai(
        messages=messages, model="gpt-3.5-turbo", temperature=0.7
    )

    assert isinstance(response, LLMResponse)
    assert response.message.role == "assistant"
    assert any(str(i) in response.message.content for i in range(1, 11))


@pytest.mark.asyncio
async def test_ask_openai_with_max_tokens():
    messages = [{"role": "user", "content": "Write a long story about a dragon"}]
    response = await ask_openai(messages=messages, model="gpt-3.5-turbo", max_tokens=20)

    assert isinstance(response, LLMResponse)
    assert len(response.message.content.split()) <= 20


@pytest.mark.asyncio
async def test_ask_openai_error():
    with pytest.raises(LLMError):
        await ask_openai(messages=[], model="non-existent-model")


# Add more test cases as needed
