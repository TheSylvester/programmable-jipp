import pytest
from pydantic import BaseModel
from llm_providers.openai_client import OpenAIClient


class WeatherResponse(BaseModel):
    temperature: float
    condition: str


@pytest.fixture
def client():
    return OpenAIClient()


def test_structured_output(client):
    weather_messages = [
        {
            "role": "system",
            "content": "You are a weather reporting assistant. Provide weather information in a structured format.",
        },
        {"role": "user", "content": "What's the weather like today in New York?"},
    ]
    response = client.generate_response(
        messages=weather_messages, output_format=WeatherResponse
    )

    assert isinstance(response, WeatherResponse)
    assert isinstance(response.temperature, float)
    assert isinstance(response.condition, str)
    assert len(response.condition) > 0


def test_regular_text_output(client):
    joke_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short joke about programming."},
    ]
    response = client.generate_response(messages=joke_messages)

    assert hasattr(response, "content")
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert "program" in response.content.lower() or "code" in response.content.lower()


def test_error_handling(client):
    with pytest.raises(Exception):
        client.generate_response(
            messages=[],  # Empty messages should raise an error
            model="non-existent-model",
        )


def test_generate_response_with_image_urls(client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you analyze these images?"},
    ]

    image_urls = [
        "https://images.squarespace-cdn.com/content/v1/60f1a490a90ed8713c41c36c/1629223610791-LCBJG5451DRKX4WOB4SP/37-design-powers-url-structure.jpeg",
        "https://www.boredpanda.com/blog/wp-content/uploads/2023/10/CvfMHtxsuYS-png__700.jpg",
    ]

    response = client.generate_response(
        messages=messages,
        image_urls=image_urls,
    )

    assert response is not None
    # Instead of checking if it's a dict, check if it has the expected attributes
    assert hasattr(response, "content")
    assert hasattr(response, "role")
    assert response.role == "assistant"


def test_generate_response_with_image_filepaths(client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you analyze these images?"},
    ]

    image_filepaths = [
        "rabbit.jpg",
        "penguin.jpg",
    ]

    response = client.generate_response(
        model="gpt-4o",
        messages=messages,
        image_filepaths=image_filepaths,
    )

    assert response is not None
    # Instead of checking if it's a dict, check if it has the expected attributes
    assert hasattr(response, "content")
    assert hasattr(response, "role")
    assert response.role == "assistant"
