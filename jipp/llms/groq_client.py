import json
import os
from typing import Union, Optional, Dict, List, Any
from openai import pydantic_function_tool
from pydantic import BaseModel
from groq import AsyncGroq
from groq.types.chat import ChatCompletion, ChatCompletionMessage
from jipp.models.jipp_models import (
    ToolCall,
    Function,
    CompletionUsage,
    LLMError,
    LLMMessage,
    LLMResponse,
    NotGiven,
    NOT_GIVEN,
)

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_MODEL = "llama-3.1-8b-instant"


async def ask_groq(
    *,
    model: str = DEFAULT_MODEL,
    messages: List[LLMMessage],
    response_format: type[BaseModel] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
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

        def groq_pydantic_function_tool(tool: BaseModel) -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": tool.model_json_schema()["title"],
                    "arguments": json.dumps(tool.model_json_schema()["properties"]),
                },
            }

        # Convert pydantic models to function tools with openai.lib._tools.pydantic_function_tool
        if tools is not NOT_GIVEN:
            processed_tools = [
                (
                    groq_pydantic_function_tool(tool)
                    if isinstance(tool, type) and issubclass(tool, BaseModel)
                    else tool
                )
                for tool in tools
            ]
        else:
            processed_tools = NOT_GIVEN

        # Handle response_format separately
        processed_response_format = NOT_GIVEN
        if response_format is not NOT_GIVEN and issubclass(response_format, BaseModel):
            # kwargs["response_format"] = pydantic_model_to_openai_schema(response_format)
            processed_response_format = {
                "type": "json_object",
            }

            # hack the system message to include the response_format
            formatting_prompt = f"You MUST respond only in this JSON Schema:\n{response_format.model_json_schema()}"

            # add response formatting prompt to the system message or...
            if messages[0].role == "system":
                messages[0].content = "\n\n".join(
                    [messages[0].content, formatting_prompt]
                )
            else:
                # ...create a new system message with the formatting prompt
                system_message = LLMMessage(
                    role="system",
                    content=f"{formatting_prompt}",
                )
                messages = [system_message] + messages

        kwargs = {
            "messages": messages,
            "model": model,
            "response_format": processed_response_format,
            "temperature": temperature,
            "top_p": top_p,
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

        # Remove NOT_GIVEN values
        kwargs = {k: v for k, v in kwargs.items() if v is not NOT_GIVEN}

        # Call Groq API
        response = await client.chat.completions.create(**kwargs)

        # Convert ChatCompletion response to LLMResponse
        return convert_chat_completion_to_llm_response(response)

    except Exception as e:
        # Convert Groq-specific errors to a generic LLMError
        raise LLMError(f"Error in Groq API call: {str(e)}")


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
