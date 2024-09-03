import pytest
from pydantic import BaseModel
from llm_providers.openai_client import OpenAIClient
from models.llm_models import Message, MessageContent, OpenAIInput


class WeatherResponse(BaseModel):
    temperature: float
    condition: str


@pytest.fixture
def client():
    return OpenAIClient()


@pytest.mark.asyncio
async def test_structured_output(client):
    weather_messages = [
        Message(
            role="system",
            content="You are a weather reporting assistant. Provide weather information in a structured format.",
        ),
        Message(role="user", content="What's the weather like today in New York?"),
    ]
    input_data = OpenAIInput(
        messages=weather_messages, model="gpt-4o-mini", response_format=WeatherResponse
    )
    response = await client.generate_response(input_data)

    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert response.content is not None
    weather_data = WeatherResponse.model_validate_json(response.content)
    assert isinstance(weather_data.temperature, float)
    assert isinstance(weather_data.condition, str)
    assert len(weather_data.condition) > 0


@pytest.mark.asyncio
async def test_regular_text_output(client):
    joke_messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Tell me a short joke about programming."),
    ]
    input_data = OpenAIInput(messages=joke_messages, model="gpt-4o-mini")
    response = await client.generate_response(input_data)

    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert "program" in response.content.lower() or "code" in response.content.lower()


@pytest.mark.asyncio
async def test_error_handling(client):
    with pytest.raises(Exception):
        await client.generate_response(
            OpenAIInput(
                messages=[],  # Empty messages should raise an error
                model="non-existent-model",
            )
        )


@pytest.mark.asyncio
async def test_generate_response_with_image_urls(client):
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(
            role="user",
            content=[
                MessageContent(type="text", text="Can you analyze these images?"),
                MessageContent(
                    type="image_url",
                    image_url={
                        "url": "https://images.squarespace-cdn.com/content/v1/60f1a490a90ed8713c41c36c/1629223610791-LCBJG5451DRKX4WOB4SP/37-design-powers-url-structure.jpeg"
                    },
                ),
                MessageContent(
                    type="image_url",
                    image_url={
                        "url": "https://www.boredpanda.com/blog/wp-content/uploads/2023/10/CvfMHtxsuYS-png__700.jpg"
                    },
                ),
            ],
        ),
    ]

    input_data = OpenAIInput(messages=messages, model="gpt-4o-mini")
    response = await client.generate_response(input_data)

    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_generate_response_with_image_filepaths(client):
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(
            role="user",
            content=[
                MessageContent(type="text", text="Can you analyze these images?"),
                MessageContent(
                    type="image_url", image_url={"filepath": "tests/rabbit.jpg"}
                ),
                MessageContent(
                    type="image_url", image_url={"filepath": "tests/penguin.jpg"}
                ),
            ],
        ),
    ]

    input_data = OpenAIInput(messages=messages, model="gpt-4o-mini")
    response = await client.generate_response(input_data)

    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
