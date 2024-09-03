from typing import Iterable, Union, Optional, Dict, List

import pydantic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat_model import ChatModel
from openai.lib._tools import pydantic_function_tool
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

from jipp.models import CompletionUsage, LLMMessage, LLMResponse


class LLMError(Exception):
    pass


async def ask_openai(
    *,
    messages: Iterable[ChatCompletionMessageParam],
    model: Union[str, ChatModel],
    response_format: Optional[type[pydantic.BaseModel]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    max_tokens: Optional[int] = None,
    stop: Union[Optional[str], List[str]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    tools: Optional[
        List[Union[ChatCompletionToolParam, type[pydantic.BaseModel]]]
    ] = None,
    tool_choice: Optional[str] = None,
    seed: Optional[int] = None,
    user: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    stream: bool = False,
    timeout: Optional[float] = None,
) -> LLMResponse:
    try:
        client = AsyncOpenAI(api_key=api_key, organization=organization)

        # Convert pydantic models to function tools with openai.lib._tools.pydantic_function_tool
        if tools:
            processed_tools = []
            for tool in tools:
                if isinstance(tool, type) and issubclass(tool, pydantic.BaseModel):
                    processed_tools.append(pydantic_function_tool(tool))
                else:
                    processed_tools.append(tool)
        else:
            processed_tools = None

        response = await client.beta.chat.completions.parse(
            messages=messages,
            model=model,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            n=n,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            tools=processed_tools,
            tool_choice=tool_choice,
            seed=seed,
            user=user,
            stream=stream,
            timeout=timeout,
        )

        return convert_parsed_chat_completion_to_llm_response(response)

    except Exception as e:
        # Convert OpenAI-specific errors to a generic LLMError
        raise LLMError(f"Error in OpenAI API call: {str(e)}")


def convert_parsed_chat_completion_to_llm_response(
    chat_completion: ParsedChatCompletion,
) -> LLMResponse:
    choice = chat_completion.choices[0]

    message_arguments = {
        "role": choice.message.role,
        "content": choice.message.content,
        "tool_calls": choice.message.tool_calls,
    }
    if choice.message.parsed:
        message_arguments["parsed"] = choice.message.parsed

    return LLMResponse(
        message=LLMMessage(**message_arguments),
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
