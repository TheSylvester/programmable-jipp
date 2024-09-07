import json
import os
from typing import Callable, Literal, Type, Union, Optional, Dict, List, Any, Tuple
from pydantic import BaseModel

import os
import asyncio
from anthropic import AsyncAnthropic
from anthropic.types.message_create_params import MessageParam, ToolParam
from anthropic.types.message import Message

import base64
import requests
from io import BytesIO

from jipp.llms.llm_selector import resolve_model_alias, get_model_profile
from jipp.models.jipp_models import (
    MessageContentImage,
    MessageContentText,
    ToolCall,
    Function,
    CompletionUsage,
    LLMError,
    LLMMessage,
    LLMResponse,
    NotGiven,
    NOT_GIVEN,
)

ClaudeModel = Literal["claude-3-5-sonnet-20240229", "claude-3-haiku-20240307"]
DEFAULT_CLAUDE_MODEL: ClaudeModel = "claude-3-5-sonnet-20240229"

client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


async def ask_claude(
    *,
    messages: List[LLMMessage],
    model: Union[str, ClaudeModel] = DEFAULT_CLAUDE_MODEL,
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
    tools: List[Type[BaseModel]] | NotGiven = NOT_GIVEN,
    tool_choice: Optional[str] | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: float | NotGiven = NOT_GIVEN,
) -> LLMResponse:
    model = resolve_model_alias(model)
    model_profile = get_model_profile(model)

    try:
        anthropic_messages, system_prompt = _convert_messages_to_anthropic(messages)
        max_tokens = (
            model_profile.max_output_tokens if max_tokens is NOT_GIVEN else max_tokens
        )

        kwargs = {
            "messages": anthropic_messages,
            "model": model,
            "max_tokens": max_tokens,
            "stop_sequences": (
                stop
                if isinstance(stop, list)
                else [stop] if stop is not NOT_GIVEN else NOT_GIVEN
            ),
            "system": system_prompt or NOT_GIVEN,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            # "metadata": {"user_id": user} if user is not NOT_GIVEN else NOT_GIVEN,
            # "stream": False,
            "timeout": timeout,
        }

        if tools is not NOT_GIVEN and model_profile.has_feature("tools"):
            kwargs["tools"] = [
                pydantic_to_anthropic_function_tool(tool) for tool in tools
            ]

        if tool_choice is not NOT_GIVEN:
            if tool_choice == "none":
                kwargs["tool_choice"] = {"type": "none"}
            elif tool_choice in ["auto", "any"]:
                kwargs["tool_choice"] = {"type": "auto"}
            else:
                kwargs["tool_choice"] = {
                    "type": "tool",
                    "name": tool_choice,
                }

        # Add response format (and prefill assistant message with "{" if provided)
        prefilled_message = None
        if response_format is not NOT_GIVEN:
            # kwargs["response_format"] = {"type": "json_object"}
            formatting_prompt = f"You MUST respond only in this JSON Schema:\n{response_format.model_json_schema()}"
            # kwargs["system"] = (
            #     kwargs.get("system", "") + f"\n\n{formatting_prompt}"
            # ).strip()

            if system_prompt:
                kwargs["system"] = system_prompt + f"\n\n{formatting_prompt}"
            else:
                kwargs["system"] = formatting_prompt
            prefilled_message = LLMMessage(role="assistant", content="{")
            kwargs["messages"] = kwargs["messages"] + [prefilled_message]

        # Remove NOT_GIVEN values
        kwargs = {k: v for k, v in kwargs.items() if v is not NOT_GIVEN}

        # Call Anthropic API
        response = await client.messages.create(**kwargs)

        # Prepend "{" to the response content if we prefilled
        if prefilled_message:
            response = prepend_brace_to_anthropic_message(response)

        # Convert Anthropic Message to LLMResponse
        return anthropic_message_to_llm_response(response)

    except Exception as e:
        # Convert Anthropic-specific errors to a generic LLMError
        raise LLMError(f"Error in Anthropic API call: {str(e)}")


def prefill_message_to_encourage_response_format(
    messages: List[LLMMessage],
) -> List[LLMMessage]:
    return messages + [LLMMessage(role="assistant", content="{")]


def pydantic_to_anthropic_function_tool(
    model: type[BaseModel],
) -> ToolParam:

    description = model.__doc__
    name = model.__name__
    input_schema = model.model_json_schema()

    anthropic_tool = {
        "input_schema": input_schema,
        "name": name,
        "description": description,
    }

    return anthropic_tool


from anthropic.types.tool_use_block_param import ToolUseBlockParam
from anthropic.types.text_block_param import TextBlockParam


def _convert_messages_to_anthropic(
    messages: List[LLMMessage],
) -> Tuple[List[Dict[str, Any]], str]:
    anthropic_messages = []
    system_prompt = ""
    if messages and messages[0].role == "system":
        system_prompt = messages[0].content
        messages = messages[1:]
    for message in messages:
        if message.role == "tool":
            anthropic_tool_result = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ],
            }
            anthropic_messages.append(anthropic_tool_result)
        else:
            content_list = []

            if isinstance(message.content, str):
                content_list.append(TextBlockParam(type="text", text=message.content))
            elif isinstance(message.content, list):
                for content in message.content:
                    if content["type"] == "text":
                        content_list.append(
                            TextBlockParam(type="text", text=content["text"])
                        )
                    elif content["type"] == "image_url":
                        image_url = content["image_url"]["url"]
                        media_type = content["image_url"].get(
                            "media_type", "image/jpeg"
                        )

                        if image_url.startswith(("http://", "https://")):
                            # Download image from URL
                            response = requests.get(image_url)
                            response.raise_for_status()
                            image_data = response.content
                            base64_data = base64.b64encode(image_data).decode("utf-8")
                        else:
                            # Assume it's already base64 encoded
                            base64_data = image_url.split(",")[1]

                        content_list.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data,
                                },
                            }
                        )

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    content_list.append(
                        ToolUseBlockParam(
                            type="tool_use",
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=json.loads(tool_call.function.arguments),
                        )
                    )

            anthropic_messages.append({"role": message.role, "content": content_list})

    return anthropic_messages, system_prompt


def convert_to_llm_message(message: Message) -> LLMMessage:
    content: List[Union[MessageContentText, MessageContentImage]] = []
    tool_calls: List[ToolCall] = []

    for block in message.content:

        block_type = block.type
        if block_type == "text":
            content.append(MessageContentText(type="text", text=block.text))
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    type="function",
                    function=Function(
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ),
                )
            )

    llm_message = LLMMessage(role="assistant", content=content)
    if tool_calls:
        llm_message.tool_calls = tool_calls

    return llm_message


def anthropic_message_to_llm_response(message: Message) -> LLMResponse:
    llm_message = convert_to_llm_message(message)

    return LLMResponse(
        message=llm_message,
        usage=(
            CompletionUsage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
            )
            if message.usage
            else None
        ),
        model=message.model,
        finish_reason=message.stop_reason,
    )


def prepend_brace_to_anthropic_message(message: Message) -> Message:
    # if isinstance(message.content, str) and message.content.strip()[-1] == "}":

    for content_block in message.content:
        if content_block.type == "text" and content_block.text.strip()[-1] == "}":
            print("\n\nDEBUG - Prepending brace to text block!\n")
            content_block.text = "{" + content_block.text
            break  # Only prepend to the first text block
        print("\n\nDEBUG - nope...\n")

    return message
