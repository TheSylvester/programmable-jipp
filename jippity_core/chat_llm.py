from jinja2 import Template
from llm_providers.llm_selector import (
    MODEL_INFO,
    get_llm_client,
    get_model_provider,
    get_model_context_window,
    is_model_supported,
)
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from models.llm_models import (
    Message,
    MessageContent,
    LLMResponse,
    OpenAIInput as OpenAIClientInput,
)
from jippity_core.token_management import trim_messages_to_context_window
import asyncio

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
DEFAULT_MODEL = "llama-3.1-8b-instant"


class AskLLMInput(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    last_response: Optional[LLMResponse] = None
    image_urls: Optional[List[str]] = None
    image_filepaths: Optional[List[str]] = None
    output_format: Optional[Any] = None
    additional_kwargs: Dict[str, Any] = {}


def render_template(template_string: str, **kwargs) -> str:
    template = Template(template_string)
    return template.render(**kwargs)


def prepare_messages(input_data: AskLLMInput, llm_client) -> List[Message]:
    messages = []
    if input_data.last_response and input_data.last_response.messages:
        messages = input_data.last_response.messages

    system = (
        render_template(input_data.system, **input_data.additional_kwargs)
        if input_data.system
        else DEFAULT_SYSTEM_PROMPT
    )
    if not messages or messages[0].role != "system":
        messages.insert(0, Message(role="system", content=system))

    user_message_content = [
        MessageContent(
            type="text",
            text=render_template(input_data.prompt, **input_data.additional_kwargs),
        )
    ]

    if input_data.image_urls:
        user_message_content.extend(
            [
                llm_client._create_image_message_from_url(url)
                for url in input_data.image_urls
            ]
        )

    if input_data.image_filepaths:
        user_message_content.extend(
            [
                llm_client._create_image_message_from_filepath(filepath)
                for filepath in input_data.image_filepaths
            ]
        )

    messages.append(Message(role="user", content=user_message_content))
    return messages


def prepare_client_input(
    messages: List[Message], input_data: AskLLMInput
) -> OpenAIClientInput:
    client_input = {
        "messages": trim_messages_to_context_window(messages, input_data.model),
        "model": input_data.model,
    }

    # Add additional kwargs only if they are provided
    for key, value in input_data.additional_kwargs.items():
        if value is not None:
            client_input[key] = value

    return OpenAIClientInput(**client_input)


async def ask_llm(input_data: AskLLMInput) -> Optional[LLMResponse]:
    try:
        if not is_model_supported(input_data.model):
            raise ValueError(f"Model {input_data.model} is not supported")

        print(f"Getting LLM client for model: {input_data.model}")
        llm_client = get_llm_client(input_data.model)

        print(f"Model provider: {get_model_provider(input_data.model)}")
        print(f"Model context window: {get_model_context_window(input_data.model)}")

        print("Preparing messages")
        messages = prepare_messages(input_data, llm_client)
        print(f"Prepared messages: {messages}")

        print("Preparing client input")
        client_input = prepare_client_input(messages, input_data)
        print(f"Prepared client input: {client_input}")

        print("Generating response")
        response_message = await llm_client.generate_response(client_input)
        print(f"Generated response: {response_message}")

        return LLMResponse(messages=messages + [response_message])
    except Exception as e:
        print(f"An error occurred in ask_llm: {e}")
        return None


async def ask_llms(
    input_data: AskLLMInput, models: List[str] = [], exclude_models: List[str] = []
) -> Dict[str, Optional[LLMResponse]]:
    async def ask_single_llm(model: str) -> tuple[str, Optional[LLMResponse]]:
        model_input = AskLLMInput(**input_data.model_dump())
        model_input.model = model
        result = await asyncio.to_thread(ask_llm, model_input)
        return model, result

    if not models:
        available_models = [
            model
            for model in MODEL_INFO.keys()
            if is_model_supported(model) and model not in exclude_models
        ]
    else:
        available_models = [
            model
            for model in models
            if is_model_supported(model) and model not in exclude_models
        ]

    tasks = [ask_single_llm(model) for model in available_models]

    results = await asyncio.gather(*tasks)
    return dict(results)
