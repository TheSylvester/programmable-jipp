import pytest
from pydantic import BaseModel, Field
from typing import List
from jipp.llms.openai_client import ask_openai, LLMError
from jipp.models.jipp_models import LLMMessage, LLMResponse


@pytest.mark.asyncio
async def test_ask_openai_basic():
    messages = [LLMMessage(role="user", content="Say 'Hello, World!'")]
    response = await ask_openai(messages=messages, model="gpt-4o-mini")

    assert isinstance(response, LLMResponse)
    assert isinstance(response.message, LLMMessage)
    assert response.message.role == "assistant"
    assert "Hello, World!" in response.message.content
    assert response.usage is not None
    assert "gpt-4o-mini" in response.model
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_ask_openai_with_temperature():
    messages = [
        LLMMessage(role="user", content="Generate a random number between 1 and 10")
    ]
    response = await ask_openai(messages=messages, model="gpt-4o-mini", temperature=0.7)

    assert isinstance(response, LLMResponse)
    assert response.message.role == "assistant"
    assert any(str(i) in response.message.content for i in range(1, 11))


@pytest.mark.asyncio
async def test_ask_openai_with_max_tokens():
    messages = [LLMMessage(role="user", content="Write a long story about a dragon")]
    response = await ask_openai(messages=messages, model="gpt-4o-mini", max_tokens=50)

    assert isinstance(response, LLMResponse)
    assert len(response.message.content.split()) <= 50
    assert response.finish_reason == "length"


@pytest.mark.asyncio
async def test_ask_openai_with_stop():
    messages = [LLMMessage(role="user", content="Count from 1 to 10")]
    response = await ask_openai(messages=messages, model="gpt-4o-mini", stop=["5"])

    assert isinstance(response, LLMResponse)
    assert "5" not in response.message.content


@pytest.mark.asyncio
async def test_ask_openai_with_tools():
    class WeatherTool(BaseModel):
        location: str = Field(
            ...,
            description="The city or location to get weather for - Full name, please",
        )
        unit: str = Field(..., description="The temperature unit, either 'C' or 'F'")

    messages = [
        LLMMessage(
            role="system",
            content="Use the WeatherTool to get the weather. Default to Fahrenheit.",
        ),
        LLMMessage(role="user", content="What's the weather in NYC?"),
    ]
    response = await ask_openai(
        messages=messages,
        model="gpt-4o-mini",
        tools=[WeatherTool],
        tool_choice="auto",
    )

    assert isinstance(response, LLMResponse)
    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) > 0
    assert response.message.tool_calls[0].function.name == "WeatherTool"
    assert "New York City" in response.message.tool_calls[0].function.arguments
    assert "F" in response.message.tool_calls[0].function.arguments


@pytest.mark.asyncio
async def test_ask_openai_with_response_format():
    class ResponseFormat(BaseModel):
        summary: str
        key_points: List[str]

    messages = [LLMMessage(role="user", content="Summarize the benefits of exercise")]
    response = await ask_openai(
        messages=messages, model="gpt-4o-mini", response_format=ResponseFormat
    )

    assert isinstance(response, LLMResponse)
    content = response.message.content
    assert "summary" in content and "key_points" in content
    # You might want to add more specific assertions here after parsing the JSON content


@pytest.mark.asyncio
async def test_ask_openai_error():
    with pytest.raises(LLMError) as exc_info:
        await ask_openai(messages=[], model="non-existent-model")

    assert "The model `non-existent-model` does not exist" in str(exc_info.value)


# Add more test cases as needed
