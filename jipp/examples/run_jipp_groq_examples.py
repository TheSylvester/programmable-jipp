import json
import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import asyncio
from typing import List
from pydantic import BaseModel, Field

from jipp.llms.groq_client import ask_groq
from jipp.models.jipp_models import LLMMessage, LLMError


async def run_ask_groq_basic():
    print("Running basic test")
    messages = [LLMMessage(role="user", content="Say 'Hello, World!'")]
    try:
        response = await ask_groq(
            messages=messages, model="llama3-groq-8b-8192-tool-use-preview"
        )
        print("Basic Response:")
        print("Role:", response.message.role)
        print("Content:", response.message.content)
        print("Model:", response.model)
        print("Finish Reason:", response.finish_reason)
        print("Usage:", response.usage)
    except Exception as e:
        print(f"Basic test failed with exception: {e}")


async def run_ask_groq_with_temperature():
    messages = [
        LLMMessage(role="user", content="Generate a random number between 1 and 10")
    ]
    try:
        response = await ask_groq(
            messages=messages,
            model="llama3-groq-8b-8192-tool-use-preview",
            temperature=0.7,
        )
        print("\nTemperature Response:")
        print("Role:", response.message.role)
        print("Content:", response.message.content)
    except Exception as e:
        print(f"Temperature test failed with exception: {e}")


async def run_ask_groq_with_max_tokens():
    messages = [LLMMessage(role="user", content="Write a long story about a dragon")]
    try:
        response = await ask_groq(
            messages=messages,
            model="llama3-groq-8b-8192-tool-use-preview",
            max_tokens=50,
        )
        print("\nMax Tokens Response:")
        print("Content:", response.message.content)
        print("Model:", response.model)
        print("Finish Reason:", response.finish_reason)
        print("Usage:", response.usage)
    except Exception as e:
        print(f"Max tokens test failed with exception: {e}")


async def run_ask_groq_with_stop():
    messages = [LLMMessage(role="user", content="Count from 1 to 10")]
    try:
        response = await ask_groq(
            messages=messages, model="llama3-groq-8b-8192-tool-use-preview", stop=["5"]
        )
        print("\nStop Sequence Response:")
        print("Content:", response.message.content)
    except Exception as e:
        print(f"Stop sequence test failed with exception: {e}")


async def run_ask_groq_with_tools():
    class WeatherTool(BaseModel):
        """Tool for getting the weather."""

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
    try:
        response = await ask_groq(
            messages=messages,
            model="llama3-groq-8b-8192-tool-use-preview",
            tools=[WeatherTool],
            tool_choice="auto",
        )
        print("\nTool Usage Response:")
        print("Content:", response.message.content)
        print("Tool Calls:", response.message.tool_calls)
    except Exception as e:
        print(f"Tool usage test failed with exception: {e}")


async def run_ask_groq_with_response_format():
    class ResponseFormat(BaseModel):
        summary: str
        key_points: List[str]

    messages = [LLMMessage(role="user", content="Summarize the benefits of exercise")]
    try:
        response = await ask_groq(
            messages=messages,
            model="llama3-groq-8b-8192-tool-use-preview",
            response_format=ResponseFormat,
        )
        print("\nResponse Format Test:")
        print("Content:", response.message.content)
        print("JSON.loads: ", json.loads(response.message.content))
        print("Parsed:", ResponseFormat.model_validate_json(response.message.content))
    except Exception as e:
        print(f"Response format test failed with exception: {e}")


async def run_ask_groq_error():
    try:
        await ask_groq(messages=[], model="non-existent-model")
    except LLMError as e:
        print("\nError test raised expected exception:", e)
    except Exception as e:
        print(f"Error test failed with unexpected exception: {e}")


async def run_ask_groq_with_images():
    from jipp.jipp_core import _create_image_message_content_from_filepath

    # test images and filepaths
    image_url_1 = "https://images.squarespace-cdn.com/content/v1/60f1a490a90ed8713c41c36c/1629223610791-LCBJG5451DRKX4WOB4SP/37-design-powers-url-structure.jpeg"
    image_url_2 = "https://www.boredpanda.com/blog/wp-content/uploads/2023/10/CvfMHtxsuYS-png__700.jpg"
    image_filepath_1 = "tests/penguin.jpg"
    image_filepath_2 = "tests/rabbit.jpg"

    messages = [
        LLMMessage(
            role="user",
            content=[
                {"type": "text", "text": "What's in each of these images?"},
                {"type": "image_url", "image_url": {"url": image_url_1}},
                {"type": "image_url", "image_url": {"url": image_url_2}},
                _create_image_message_content_from_filepath(image_filepath_1),
                _create_image_message_content_from_filepath(image_filepath_2),
            ],
        )
    ]

    try:
        response = await ask_groq(
            messages=messages, model="llama3-groq-8b-8192-tool-use-preview"
        )
        print("\nImage Inference Response:")
        print("Content:", response.message.content)
    except Exception as e:
        print(f"Image inference test failed with exception: {e}")


async def run_all_tests():
    # await run_ask_groq_basic()
    # await run_ask_groq_with_temperature()
    # await run_ask_groq_with_max_tokens()
    # await run_ask_groq_with_stop()
    await run_ask_groq_with_tools()
    # await run_ask_groq_with_response_format()
    # await run_ask_groq_error()
    # await run_ask_groq_with_images()


def main():
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
