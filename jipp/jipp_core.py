import base64
import io
from PIL import Image
from jinja2 import Template
from jipp.llms.llm_selector import get_llm_info
from jipp.models.jipp_models import (
    Conversation,
    ImageURL,
    LLMMessage,
    LLMResponse,
    MessageContentImage,
    Tool,
    ToolCall,
)
from llm_providers.llm_selector import (
    is_model_supported,
)
from typing import Literal, Optional, List, Dict, Union
from pydantic import BaseModel
from openai._types import NotGiven, NOT_GIVEN
from jippity_core.token_management import trim_messages_to_context_window


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
DEFAULT_MODEL = "llama-3.1-8b-instant"


def render_template(template_string: str, **kwargs) -> str:
    template = Template(template_string)
    return template.render(**kwargs)


def _create_image_message_content_from_url(url: str) -> Dict[str, Dict[str, str]]:
    return MessageContentImage(
        type="image_url", image_url=ImageURL(url=url, detail="auto")
    )


def _create_image_message_content_from_filepath(
    filepath: str,
) -> Dict[str, Dict[str, str]]:
    with open(filepath, "rb") as image_file:
        image_bytes = io.BytesIO(image_file.read())
        image = Image.open(image_bytes)
        format = image.format
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        url = f"data:image/{format.lower()};base64,{base64_image}"

        return MessageContentImage(
            type="image_url", image_url=ImageURL(url=url, detail="auto")
        )


def process_image_inputs_into_contents(
    image_urls: List[str], image_filepaths: List[str]
) -> List[BaseModel]:

    user_message_content = []
    if image_urls:
        user_message_content.extend(
            [_create_image_message_content_from_url(url) for url in image_urls]
        )

    if image_filepaths:
        user_message_content.extend(
            [
                _create_image_message_content_from_filepath(filepath)
                for filepath in image_filepaths
            ]
        )

    return user_message_content


def render_template(template_string: str, **kwargs) -> str:
    template = Template(template_string)
    return template.render(**kwargs)


async def ask_llm(
    *,
    model: str,
    prompt: str,
    system: Optional[str] = None,
    conversation: Optional[Conversation] = None,
    response_format: type[BaseModel] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
    n: Optional[int] | NotGiven = NOT_GIVEN,
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    tools: List[Tool] | NotGiven = NOT_GIVEN,
    tool_choice: Optional[str] | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    images: List[Dict[Literal["url", "filepath"], str]] | NotGiven = NOT_GIVEN,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: float | NotGiven = NOT_GIVEN,
    **variables,
) -> LLMResponse:

    kwargs = {
        "model": model,
        "response_format": response_format,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
        "stop": stop,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "tool_choice": tool_choice,
        "seed": seed,
        "user": user,
        "timeout": timeout,
    }

    # Remove NOT_GIVEN values
    kwargs = {k: v for k, v in kwargs.items() if v is not NOT_GIVEN}

    # pass tool definitions only, the tool functions are for us to execute
    if tools is not NOT_GIVEN:
        kwargs["tools"] = [tool.definition for tool in tools]

    try:
        if not is_model_supported(model):
            raise ValueError(f"Model {model} is not supported")

        print(f"Getting LLM info for model: {model}")
        llm_info = get_llm_info(model)
        ask_client = llm_info.client

        messages = [m for m in conversation.messages] if conversation else []

        if system:
            system_prompt = render_template(system, **variables)
            system_message = LLMMessage(role="system", content=system_prompt)

            # remove the first message if it's a system message
            if messages and messages[0].role == "system":
                messages = messages[1:]
            messages.insert(0, system_message)

        user_message = LLMMessage(
            role="user", content=render_template(prompt, **variables)
        )
        messages.append(user_message)

        print(f"Prepared messages: {messages}")

        print("Generating response")
        response = await ask_client(messages, **kwargs)

        print(f"Got response: {response}")
        # did we get a tool call?
        while response.tool_calls:
            print(f"Got tool calls: {response.tool_calls}")
            for tool_call in response.tool_calls:
                id = tool_call.id
                tool_response = await execute_tool_call(tool_call, tools)
                print(f"Tool response: {tool_response}")
                tool_message = LLMMessage(
                    tool_call_id=id, role="tool", content=tool_response
                )
                messages.append(tool_message)
            response = await ask_client(messages, **kwargs)
            print(f"Got response: {response}")

        # parse output if we have a response model
        add_parsed = {}
        try:
            if response_format is not NOT_GIVEN and issubclass(
                response_format, BaseModel
            ):
                parsed_response = response_format.model_validate_json(
                    response.message.content
                )
                add_parsed["parsed"] = parsed_response
        except Exception as e:
            print(
                f"WARNING: Parsing error from output to response_format: {e} - output format IGNORED"
            )

        return Conversation(
            messages=messages + [response.message],
            usage=response.usage,
            model=response.model,
            finish_reason=response.finish_reason,
            **add_parsed,
        )

    except Exception as e:
        print(f"An error occurred in ask_llm: {e}")
        return None


async def execute_tool_call(tool_call: ToolCall, tools: List[Tool]) -> str:
    """
    Execute a tool call selected from the list of tools.
    """
    for tool in tools:
        if tool.name == tool_call.function.name:
            validated_args = tool.definition.model_validate_json(
                tool_call.function.arguments
            )
            arguments = validated_args.model_dump()
            return str(await tool(**arguments))

    raise ValueError(f"Tool {tool_call.function.name} not found")
