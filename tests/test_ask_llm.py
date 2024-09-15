import pytest
from pydantic import BaseModel, Field
from typing import List

from jipp.jipp_engine import ask_llm
from jipp.models.jipp_models import Conversation, LLMError, Tool


@pytest.mark.asyncio
async def test_ask_llm_basic():
    conversation = await ask_llm(
        model="gpt-4o-mini",
        prompt="Say 'Hello, World!'",
        system="You are a helpful assistant.",
    )

    assert isinstance(conversation, Conversation)
    assert len(conversation) > 0
    assert "Hello, World!" in str(conversation)
    assert conversation.usage is not None
    assert "gpt-4o-mini" in conversation.model
    assert conversation.finish_reason == "stop"


@pytest.mark.asyncio
async def test_ask_llm_with_temperature():
    conversation = await ask_llm(
        model="gpt-4o-mini",
        prompt="Generate a random number between 1 and 10",
        system="You are a helpful assistant.",
        temperature=0.7,
    )

    assert isinstance(conversation, Conversation)
    assert any(str(i) in str(conversation) for i in range(1, 11))


@pytest.mark.asyncio
async def test_ask_llm_with_max_tokens():
    conversation = await ask_llm(
        model="gpt-4o-mini",
        prompt="Write a long story about a dragon",
        system="You are a helpful assistant.",
        max_tokens=50,
    )

    assert isinstance(conversation, Conversation)
    assert len(str(conversation).split()) <= 50
    assert conversation.finish_reason == "length"


@pytest.mark.asyncio
async def test_ask_llm_with_stop():
    conversation = await ask_llm(
        model="gpt-4o-mini",
        prompt="Count from 1 to 10",
        system="You are a helpful assistant.",
        stop=["5"],
    )

    assert isinstance(conversation, Conversation)
    assert "5" not in str(conversation)


@pytest.mark.asyncio
async def test_ask_llm_with_tools():
    class WeatherTool(BaseModel):
        """Tool for getting the weather."""

        location: str = Field(
            ...,
            description="The city or location to get weather for - Full name, please",
        )
        unit: str = Field(..., description="The temperature unit, either 'C' or 'F'")

    async def get_weather(location: str, unit: str) -> str:
        # This is a mock function. In a real scenario, you'd call a weather API here.
        return f"The weather in {location} is 72°{unit}."

    conversation = await ask_llm(
        model="gpt-4o-mini",
        prompt="What's the weather in NYC?",
        system="Use the WeatherTool to get the weather. Default to Fahrenheit.",
        tools=[{"schema": WeatherTool, "function": get_weather}],
        tool_choice="auto",
    )

    assert isinstance(conversation, Conversation)
    assert len(conversation) > 0
    system_message = conversation[0]
    assert system_message.role == "system"
    user_message = conversation[1]
    assert user_message.role == "user"
    assert user_message.content == "What's the weather in NYC?"
    tool_call_message = conversation[-3]
    assert tool_call_message.tool_calls is not None
    assert tool_call_message.tool_calls[0].function.name == "WeatherTool"
    assert "New York City" in tool_call_message.tool_calls[0].function.arguments
    assert "F" in tool_call_message.tool_calls[0].function.arguments


@pytest.mark.asyncio
async def test_ask_llm_with_response_format():
    class ResponseFormat(BaseModel):
        summary: str
        key_points: List[str]

    conversation = await ask_llm(
        model="gpt-4o-mini",
        prompt="Summarize the benefits of exercise",
        system="You are a helpful assistant.",
        response_format=ResponseFormat,
    )

    assert isinstance(conversation, Conversation)
    assert conversation.parsed is not None
    assert isinstance(conversation.parsed, ResponseFormat)
    assert hasattr(conversation.parsed, "summary")
    assert hasattr(conversation.parsed, "key_points")
    assert isinstance(conversation.parsed.key_points, list)


@pytest.mark.asyncio
async def test_ask_llm_error():
    with pytest.raises(LLMError) as exc_info:
        await ask_llm(model="non-existent-model", prompt="This should fail")

    assert "Model non-existent-model is not supported" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ask_llm_with_images():
    image_url_1 = "https://images.squarespace-cdn.com/content/v1/60f1a490a90ed8713c41c36c/1629223610791-LCBJG5451DRKX4WOB4SP/37-design-powers-url-structure.jpeg"
    image_filepath_1 = "tests/rabbit.jpg"

    conversation = await ask_llm(
        model="gpt-4o",
        prompt="What's in each of these images?",
        system="You are a helpful assistant.",
        images=[
            {"url": image_url_1},
            {"filepath": image_filepath_1},
        ],
    )

    assert isinstance(conversation, Conversation)
    assert len(conversation) > 0
    content = str(conversation).lower()

    # Check if the response mentions key elements from each image
    assert "target" in content or "bullseye" in content
    assert (
        "rabbit" in content
        or "bunny" in content
        or "animal" in content
        or "toy" in content
    )

    assert conversation.usage is not None
    assert "gpt-4o" in conversation.model
    assert conversation.finish_reason == "stop"
