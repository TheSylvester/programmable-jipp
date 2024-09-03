import asyncio
from pydantic import BaseModel
from jipp import ChatLLM, AskLLMInput, Tool


class MathResponse(BaseModel):
    result: int


class AddNumbers(BaseModel):
    a: int
    b: int


async def main():
    # Initialize the ChatLLM
    llm = ChatLLM()

    # Define a simple tool
    tools = [
        Tool(
            name="add_numbers",
            description="Add two numbers",
            parameters=AddNumbers,
            function=lambda a, b: a + b,
        )
    ]

    # Prepare the input
    input_data = AskLLMInput(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        tools=tools,
        response_format=MathResponse,
    )

    # Generate a response
    response = await llm.ask_llm(input_data)

    print(f"Raw response: {response.raw_response}")
    print(f"Parsed content: {response.parsed_content}")


if __name__ == "__main__":
    asyncio.run(main())
