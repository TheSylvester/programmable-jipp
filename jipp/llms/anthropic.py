from typing import List, Dict, Any, Union
import asyncio
import base64
import requests
from anthropic import Anthropic
from pydantic import BaseModel
from jipp.llms.base import BaseLLMClient
from jipp.models.requests import AskLLMInput
from jipp.models.responses import LLMResponse
from jipp.config.settings import settings
from jipp.utils import log_exceptions


class AnthropicClient(BaseLLMClient):
    def __init__(self):
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.system = None

    @log_exceptions
    async def generate_response(self, input_data: AskLLMInput) -> LLMResponse:
        try:
            anthropic_messages = self._convert_messages(input_data.messages)

            request_params = {
                "model": input_data.model,
                "messages": anthropic_messages,
                "temperature": input_data.temperature,
                "max_tokens": input_data.max_tokens,
                "stream": input_data.stream,
            }

            if self.system:
                request_params["system"] = self.system

            if input_data.tools:
                request_params["tools"] = self._convert_tools_to_openai_format(
                    input_data.tools
                )
                request_params["tool_choice"] = input_data.additional_kwargs.get(
                    "tool_choice", "auto"
                )

            response = await asyncio.to_thread(
                self.client.messages.create, **request_params
            )

            content = self._process_response_content(response.content)

            return LLMResponse(
                raw_response=response,
                parsed_content=content,
            )
        except Exception as error:
            print(f"An error occurred in AnthropicClient: {str(error)}")
            raise

    def _convert_messages(self, messages: List[dict]) -> List[Dict[str, Any]]:
        anthropic_messages = []
        if messages and messages[0]["role"] == "system":
            self.system = messages[0]["content"]
            messages = messages[1:]
        for message in messages:
            if isinstance(message["content"], str):
                anthropic_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )
            else:
                content_list = []
                for content in message["content"]:
                    if content["type"] == "text":
                        content_list.append({"type": "text", "text": content["text"]})
                    elif content["type"] == "image_url":
                        content_list.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": content["image_url"].get(
                                        "media_type", "image/jpeg"
                                    ),
                                    "data": content["image_url"]["url"].split(",")[
                                        1
                                    ],  # Assuming base64 data
                                },
                            }
                        )
                anthropic_messages.append(
                    {"role": message["role"], "content": content_list}
                )
        return anthropic_messages

    def _process_response_content(
        self, content: List[Any]
    ) -> Union[str, List[Dict[str, Any]]]:
        if len(content) == 1 and content[0].type == "text":
            return content[0].text
        else:
            return [{"type": item.type, "text": item.text} for item in content]

    def _create_image_message_from_url(self, url: str) -> dict:
        encoded_image = self._download_and_encode_image(url)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        }

    def _create_image_message_from_filepath(self, filepath: str) -> dict:
        with open(filepath, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        }

    def _download_and_encode_image(self, url: str) -> str:
        response = requests.get(url)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            raise ValueError(f"Failed to download image from {url}")
