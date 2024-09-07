import os
from typing import Union, Optional, Dict, List, Any, Tuple
from pydantic import BaseModel
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
)
from openai.types.chat import ChatCompletionMessage
from openai.types.chat_model import ChatModel
from openai._types import NotGiven, NOT_GIVEN
from openai.lib._tools import pydantic_function_tool

from jipp.llms.pydantic_to_schema import pydantic_model_to_openai_schema
from jipp.models.jipp_models import (
    ToolCall,
    Function,
    CompletionUsage,
    LLMError,
    LLMMessage,
    LLMResponse,
)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def ask_openai(
    *,
    model: Union[str, ChatModel],
    messages: List[LLMMessage],
    response_format: type[BaseModel] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
    top_k: Optional[int] | NotGiven = NOT_GIVEN,
    n: Optional[int] | NotGiven = NOT_GIVEN,
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    tools: List[BaseModel] | NotGiven = NOT_GIVEN,
    tool_choice: Optional[str] | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: float | NotGiven = NOT_GIVEN,
) -> LLMResponse:
    try:

        # Convert pydantic models to function tools with openai.lib._tools.pydantic_function_tool
        if tools is not NOT_GIVEN:
            processed_tools = [
                (
                    pydantic_function_tool(tool)
                    if isinstance(tool, type) and issubclass(tool, BaseModel)
                    else tool
                )
                for tool in tools
            ]
        else:
            processed_tools = NOT_GIVEN

        kwargs = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n": n,
            "max_tokens": max_tokens,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "tools": processed_tools,
            "tool_choice": tool_choice,
            "seed": seed,
            "user": user,
            "timeout": timeout,
        }

        # Handle response_format separately
        if response_format is not NOT_GIVEN:
            kwargs["response_format"] = pydantic_model_to_openai_schema(response_format)

        # Remove NOT_GIVEN values
        kwargs = {k: v for k, v in kwargs.items() if v is not NOT_GIVEN}

        # Call OpenAI API
        response = await client.chat.completions.create(**kwargs)

        # Convert ChatCompletion response to LLMResponse
        return convert_chat_completion_to_llm_response(response)

    except Exception as e:
        # Convert OpenAI-specific errors to a generic LLMError
        raise LLMError(f"Error in OpenAI API call: {str(e)}")


def convert_to_llm_message(
    message: ChatCompletionMessage,
) -> LLMMessage:

    llm_message = LLMMessage(role="assistant", content=message.content)

    if message.function_call:
        llm_message.function_call = Function(
            name=message.function_call.name, arguments=message.function_call.arguments
        )

    if message.tool_calls:
        llm_message.tool_calls = [
            ToolCall(
                id=tool_call.id,
                type=tool_call.type,
                function=Function(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for tool_call in message.tool_calls
        ]

    return llm_message


def convert_chat_completion_to_llm_response(
    chat_completion: ChatCompletion,
) -> LLMResponse:

    choice = chat_completion.choices[0]
    message = convert_to_llm_message(choice.message)

    return LLMResponse(
        message=message,
        usage=(
            CompletionUsage(
                prompt_tokens=chat_completion.usage.prompt_tokens,
                completion_tokens=chat_completion.usage.completion_tokens,
                total_tokens=chat_completion.usage.total_tokens,
            )
            if chat_completion.usage
            else None
        ),
        model=chat_completion.model,
        finish_reason=choice.finish_reason,
    )
