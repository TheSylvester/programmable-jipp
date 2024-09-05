import asyncio
from typing import List
from pydantic import BaseModel, Field
from jipp.jipp_core import ask_llm
from jipp.models.jipp_models import LLMError


async def run_ask_llm_basic():
    try:
        conversation = await ask_llm(
            model="gpt-4o-mini",
            prompt="Say 'Hello, World!'",
            system="You are a helpful assistant.",
        )
        print("Basic Response:")
        print("Content:", str(conversation))
        print("Model:", conversation.model)
        print("Finish Reason:", conversation.finish_reason)
        print("Usage:", conversation.usage)
    except Exception as e:
        print(f"Basic test failed with exception: {e}")


async def run_ask_llm_with_temperature():
    try:
        conversation = await ask_llm(
            model="gpt-4o-mini",
            prompt="Generate a random number between 1 and 10",
            system="You are a helpful assistant.",
            temperature=0.7,
        )
        print("\nTemperature Response:")
        print("Content:", str(conversation))
    except Exception as e:
        print(f"Temperature test failed with exception: {e}")


async def run_ask_llm_with_max_tokens():
    try:
        conversation = await ask_llm(
            model="gpt-4o-mini",
            prompt="Write a long story about a dragon",
            system="You are a helpful assistant.",
            max_tokens=50,
        )
        print("\nMax Tokens Response:")
        print("Content:", str(conversation))
        print("Model:", conversation.model)
        print("Finish Reason:", conversation.finish_reason)
        print("Usage:", conversation.usage)
    except Exception as e:
        print(f"Max tokens test failed with exception: {e}")


async def run_ask_llm_with_stop():
    try:
        conversation = await ask_llm(
            model="gpt-4o-mini",
            prompt="Count from 1 to 10",
            system="You are a helpful assistant.",
            stop=["5"],
        )
        print("\nStop Sequence Response:")
        print("Content:", str(conversation))
    except Exception as e:
        print(f"Stop sequence test failed with exception: {e}")


async def run_ask_llm_with_tools():
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

    try:
        conversation = await ask_llm(
            model="gpt-4o-mini",
            prompt="What's the weather in NYC?",
            system="Use the WeatherTool to get the weather. Default to Fahrenheit.",
            tools=[{"schema": WeatherTool, "function": get_weather}],
            tool_choice="auto",
        )
        print("\nTool Usage Response:")
        print("Content:", str(conversation))
        print("Tool Calls:", conversation[-3].tool_calls)
    except Exception as e:
        print(f"Tool usage test failed with exception: {e}")


async def run_ask_llm_with_response_format():
    class ResponseFormat(BaseModel):
        summary: str
        key_points: List[str]

    try:
        conversation = await ask_llm(
            model="gpt-4o-mini",
            prompt="Summarize the benefits of exercise",
            system="You are a helpful assistant.",
            response_format=ResponseFormat,
        )
        print("\nResponse Format Test:")
        print("Content:", str(conversation))
        if conversation.parsed:
            print("Parsed Response:", conversation.parsed)
    except Exception as e:
        print(f"Response format test failed with exception: {e}")


async def run_ask_llm_error():
    try:
        await ask_llm(model="non-existent-model", prompt="This should fail")
    except LLMError as e:
        print("\nError test raised expected exception:", e)
    except Exception as e:
        print(f"Error test failed with unexpected exception: {e}")


async def run_ask_llm_with_images():
    image_url_1 = "https://images.squarespace-cdn.com/content/v1/60f1a490a90ed8713c41c36c/1629223610791-LCBJG5451DRKX4WOB4SP/37-design-powers-url-structure.jpeg"
    image_filepath_1 = "tests/rabbit.jpg"

    try:
        conversation = await ask_llm(
            model="gpt-4o",
            prompt="What's in each of these images?",
            system="You are a helpful assistant.",
            images=[
                {"url": image_url_1},
                {"filepath": image_filepath_1},
            ],
        )
        print("\nImage Inference Response:")
        print("Content:", str(conversation))
    except Exception as e:
        print(f"Image inference test failed with exception: {e}")


async def run_all_tests():
    await run_ask_llm_basic()
    # await run_ask_llm_with_temperature()
    # await run_ask_llm_with_max_tokens()
    # await run_ask_llm_with_stop()
    # await run_ask_llm_with_tools()
    # await run_ask_llm_with_response_format()
    # await run_ask_llm_error()
    # await run_ask_llm_with_images()


def main():
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
