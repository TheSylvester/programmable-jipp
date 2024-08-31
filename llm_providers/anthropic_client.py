import os
import base64
from typing import List, Optional, Dict, Any, Union
from anthropic import Anthropic
from .llm_client import LLMClient
from models.llm_models import Message, MessageContent, AnthropicInput
from error_handlers.anthropic_error import handle_anthropic_error


class AnthropicClient(LLMClient):
    def __init__(self):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.system = None

    def generate_response(self, input_data: AnthropicInput) -> Message:
        try:
            # Convert messages to the format expected by Anthropic
            anthropic_messages = self._convert_messages(input_data.messages)

            # Prepare the request parameters
            request_params = {
                "model": input_data.model,
                "messages": anthropic_messages,
            }

            # Add optional parameters if provided
            optional_params = [
                "temperature",
                "top_k",
                "top_p",
                "metadata",
                "stop_sequences",
                "stream",
            ]
            for param in optional_params:
                value = getattr(input_data, param, None)
                if value is not None:
                    request_params[param] = value

            # Add system message if available
            if self.system:
                request_params["system"] = self.system

            # Handle max_tokens parameter
            if input_data.max_tokens is not None:
                request_params["max_tokens"] = input_data.max_tokens
            else:
                # If max_tokens is not provided, use the model's max tokens
                from llm_selector import get_max_tokens  # Lazy import

                request_params["max_tokens"] = get_max_tokens(input_data.model)

            # Add tools if provided
            if input_data.tools:
                request_params["tools"] = input_data.tools
                request_params["tool_choice"] = input_data.tool_choice or "auto"

            response = self.client.messages.create(**request_params)

            # Handle potential multiple content blocks in the response
            content = self._process_response_content(response.content)

            return Message(
                role="assistant",
                content=content,
            )
        except Exception as error:
            handle_anthropic_error(error)
            return Message(
                role="error",
                content=f"An error occurred: {str(error)}",
            )

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        anthropic_messages = []
        print(f"Converting messages *pre*: {messages}")
        if messages and messages[0].role == "system":
            self.system = messages[0].content
            messages = messages[1:]
        print(f"Converting messages: {messages}")
        for message in messages:
            if isinstance(message.content, str):
                anthropic_messages.append(
                    {"role": message.role, "content": message.content}
                )
            else:
                content_list = []
                for content in message.content:
                    if content.type == "text":
                        content_list.append({"type": "text", "text": content.text})
                    elif content.type == "image_url":
                        content_list.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": content.image_url.get(
                                        "media_type", "image/jpeg"
                                    ),
                                    "data": content.image_url["url"].split(",")[
                                        1
                                    ],  # Assuming base64 data
                                },
                            }
                        )
                anthropic_messages.append(
                    {"role": message.role, "content": content_list}
                )
        return anthropic_messages

    def _process_response_content(
        self, content: List[Any]
    ) -> Union[str, List[Dict[str, Any]]]:
        if len(content) == 1 and content[0].type == "text":
            return content[0].text
        else:
            return [{"type": item.type, "text": item.text} for item in content]

    def _create_image_message_from_url(self, url: str) -> MessageContent:
        encoded_image = self._download_and_encode_image(url)
        return MessageContent(
            type="image_url",
            image_url={"url": f"data:image/jpeg;base64,{encoded_image}"},
        )

    def _create_image_message_from_filepath(self, filepath: str) -> MessageContent:
        with open(filepath, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return MessageContent(
            type="image_url",
            image_url={"url": f"data:image/jpeg;base64,{encoded_image}"},
        )

    def _download_and_encode_image(self, url: str) -> str:
        import requests

        response = requests.get(url)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            raise ValueError(f"Failed to download image from {url}")
