from typing import Dict, Optional, List, Any
from jinja2 import Template
from jipp.llms.base import BaseLLMClient
from jipp.llms.openai import OpenAIClient
from jipp.llms.anthropic import AnthropicClient
from jipp.llms.groq import GroqClient
from jipp.models.requests import AskLLMInput
from jipp.models.responses import LLMResponse
from jipp.models.llm_models import (
    Message,
    MessageContent,
    OpenAIInput as OpenAIClientInput,
)
from jipp.token_management import (
    trim_messages_to_context_window,
    count_tokens_in_messages,
)
from jipp.llms.llm_selector import get_model_context_window
import asyncio

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
DEFAULT_MODEL = "llama-3.1-8b-instant"


class ChatLLM:
    """
    A class that provides a unified interface to interact with various LLM providers.

    This class manages different LLM clients and routes requests to the appropriate
    client based on the specified model.
    """

    def __init__(self):
        """
        Initialize the ChatLLM with available LLM clients.
        """
        self.clients: Dict[str, BaseLLMClient] = {
            "openai": OpenAIClient(),
            "groq": GroqClient(),
            "anthropic": AnthropicClient(),
        }

    async def ask_llm(self, input_data: AskLLMInput) -> Optional[LLMResponse]:
        """
        Send a request to the appropriate LLM based on the input model.

        Args:
            input_data (AskLLMInput): The input data containing the model and other parameters.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            ValueError: If an unsupported model provider is specified.
        """
        try:
            provider = self._get_provider(input_data.model)
            client = self.clients.get(provider)
            if not client:
                raise ValueError(f"Unsupported model provider: {provider}")

            messages = self.prepare_messages(input_data, client)
            client_input = self.prepare_client_input(messages, input_data)
            response_message = await client.generate_response(client_input)

            return LLMResponse(messages=messages + [response_message])
        except Exception as e:
            print(f"An error occurred in ask_llm: {e}")
            return None

    def _get_provider(self, model: str) -> str:
        """
        Determine the provider based on the model name.

        Args:
            model (str): The name of the model.

        Returns:
            str: The name of the provider.

        Raises:
            ValueError: If the model provider cannot be determined.
        """
        model_lower = model.lower()
        if model_lower.startswith("gpt"):
            return "openai"
        elif model_lower.startswith("claude"):
            return "anthropic"
        elif model_lower.startswith("llama") or model_lower.startswith("mixtral"):
            return "groq"
        else:
            raise ValueError(f"Unknown model provider for model: {model}")

    def render_template(self, template_string: str, **kwargs) -> str:
        template = Template(template_string)
        return template.render(**kwargs)

    def prepare_messages(self, input_data: AskLLMInput, llm_client) -> List[Message]:
        messages = []
        if input_data.last_response and input_data.last_response.messages:
            messages = input_data.last_response.messages

        system = (
            self.render_template(input_data.system, **input_data.additional_kwargs)
            if input_data.system
            else DEFAULT_SYSTEM_PROMPT
        )
        if not messages or messages[0].role != "system":
            messages.insert(0, Message(role="system", content=system))

        user_message_content = [
            MessageContent(
                type="text",
                text=self.render_template(
                    input_data.prompt, **input_data.additional_kwargs
                ),
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
        self, messages: List[Message], input_data: AskLLMInput
    ) -> OpenAIClientInput:
        context_window = get_model_context_window(input_data.model)
        trimmed_messages = trim_messages_to_context_window(messages, input_data.model)

        client_input = {
            "messages": trimmed_messages,
            "model": input_data.model,
        }

        # Add additional kwargs only if they are provided
        for key, value in input_data.additional_kwargs.items():
            if value is not None:
                client_input[key] = value

        return OpenAIClientInput(**client_input)

    async def ask_llms(
        self,
        input_data: AskLLMInput,
        models: List[str] = [],
        exclude_models: List[str] = [],
    ) -> Dict[str, Optional[LLMResponse]]:
        async def ask_single_llm(model: str) -> tuple[str, Optional[LLMResponse]]:
            model_input = AskLLMInput(**input_data.model_dump())
            model_input.model = model
            result = await asyncio.to_thread(self.ask_llm, model_input)
            return model, result

        if not models:
            available_models = [
                model for model in self.clients.keys() if model not in exclude_models
            ]
        else:
            available_models = [
                model for model in models if model not in exclude_models
            ]

        tasks = [ask_single_llm(model) for model in available_models]

        results = await asyncio.gather(*tasks)
        return dict(results)
