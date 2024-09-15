import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import asyncio
from typing import List
from pydantic import BaseModel, Field

from jipp.llms.anthropic_client import ask_claude
from jipp.models.jipp_models import LLMMessage, LLMError


async def run_ask_claude_basic():
    messages = [LLMMessage(role="user", content="Say 'Hello, World!'")]
    try:
        response = await ask_claude(messages=messages, model="claude-sonnet")
        print("Basic Response:")
        print("Role:", response.message.role)
        print("Content:", response.message.content)
        print("Model:", response.model)
        print("Finish Reason:", response.finish_reason)
        print("Usage:", response.usage)
    except Exception as e:
        print(f"Basic test failed with exception: {e}")


async def run_ask_claude_with_temperature():
    query = """Follow instructions carefully, especially with spelling.List the 5 most commonly used words starting with F.
MAKE SURE THE WORDS START WITH F"""
    # model =     "claude-3-5-sonnet-20240620"
    model = "claude-haiku"
    messages = [LLMMessage(role="user", content=query)]
    temperatures = [0.0, 1.0]  # List of temperatures to test
    for temp in temperatures:
        try:
            # highest temperatures do not answer with F properly on haiku
            response = await ask_claude(
                messages=messages, model=model, temperature=temp
            )
            print(f"\nTemperature Response (temperature={temp}):")
            print("Role:", response.message.role)
            print("Content:", "\n", response.message.content[0]["text"])
            for line in response.message.content[0]["text"].split("\n"):
                # check if the line starts with a number followed by a period and does not contain "F"
                # should happen on highest temperatures, should not happen on lowest temperatures
                if ". " in line and not ". F".lower() in line.lower():
                    print("\n!! mismatch !!\n", line, "\n")

        except Exception as e:
            print(f"Temperature test failed with exception for temperature {temp}: {e}")


async def run_ask_claude_with_max_tokens():
    messages = [LLMMessage(role="user", content="Write a long story about a dragon")]
    try:
        response = await ask_claude(
            messages=messages, model="claude-3-5-sonnet-20240620", max_tokens=50
        )
        print("\nMax Tokens Response:")
        print("Content:", response.message.content)
        print("Model:", response.model)
        print("Finish Reason:", response.finish_reason)
        print("Usage:", response.usage)
    except Exception as e:
        print(f"Max tokens test failed with exception: {e}")


async def run_ask_claude_with_stop():
    messages = [LLMMessage(role="user", content="Count from 1 to 10")]
    try:
        response = await ask_claude(
            messages=messages, model="claude-3-5-sonnet-20240620", stop=["5"]
        )
        print("\nStop Sequence Response:")
        print("Content:", response.message.content)
    except Exception as e:
        print(f"Stop sequence test failed with exception: {e}")


async def run_ask_claude_with_tools():
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
        response = await ask_claude(
            messages=messages,
            model="claude-haiku",
            tools=[WeatherTool],
            tool_choice="auto",
        )
        print("\nTool Usage Response:", response)
        print("Content:", response.message.content)
        print("Tool Calls:", response.message.tool_calls)
    except Exception as e:
        print(f"Tool usage test failed with exception: {e}")


async def run_ask_claude_with_response_format():
    class ResponseFormat(BaseModel):
        summary: str
        key_points: List[str]

    messages = [LLMMessage(role="user", content="Summarize the benefits of exercise")]
    try:
        response = await ask_claude(
            messages=messages,
            model="claude-haiku",
            response_format=ResponseFormat,
        )
        print("\nResponse Format Test:")
        print("Content:", response.message.content[0]["text"])
        # You would typically parse this content as JSON and validate against ResponseFormat

    except Exception as e:
        print(f"Response format test failed with exception: {e}")


async def run_ask_claude_error():
    try:
        await ask_claude(messages=[], model="non-existent-model")
    except LLMError as e:
        print("\nError test raised expected exception:", e)
    except Exception as e:
        print(f"Error test failed with unexpected exception: {e}")


async def run_ask_claude_with_images():
    from jipp.jipp_engine import _create_image_message_content_from_filepath

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
                _create_image_message_content_from_filepath(image_filepath_1),
                {"type": "image_url", "image_url": {"url": image_url_1}},
                # {"type": "image_url", "image_url": {"url": image_url_2}},
                # _create_image_message_content_from_filepath(image_filepath_2),
            ],
        )
    ]

    try:
        response = await ask_claude(
            messages=messages, model="claude-3-5-sonnet-20240620"
        )
        print("\nImage Inference Response:")
        print("Content:", response.message.content[0]["text"])
    except Exception as e:
        print(f"Image inference test failed with exception: {e}")


async def run_all_tests():
    await run_ask_claude_basic()
    await run_ask_claude_with_temperature()
    await run_ask_claude_with_max_tokens()
    await run_ask_claude_with_stop()
    await run_ask_claude_with_tools()
    await run_ask_claude_with_response_format()
    await run_ask_claude_error()
    await run_ask_claude_with_images()


def main():
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
