from typing import List, Dict, Any
import asyncio
from groq import Groq
from pydantic import BaseModel
from jipp.llms.base import BaseLLMClient
from jipp.models.requests import AskLLMInput
from jipp.models.responses import LLMResponse
from jipp.config.settings import settings
from jipp.utils import log_exceptions


class GroqClient(BaseLLMClient):
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    @log_exceptions
    async def generate_response(self, input_data: AskLLMInput) -> LLMResponse:
        try:
            # Process tools if any
            if input_data.tools:
                input_data.additional_kwargs["tool_choice"] = (
                    input_data.additional_kwargs.get("tool_choice", "auto")
                )

            # Process and create image content
            self._process_image_content(input_data)

            # Convert Pydantic model to dict, excluding None values
            input_dict = input_data.model_dump(exclude_none=True)

            # Convert messages to the format expected by Groq (OpenAI-compatible)
            input_dict["messages"] = [
                {
                    "role": m["role"],
                    "content": (
                        m["content"]
                        if isinstance(m["content"], str)
                        else [c for c in m["content"]]
                    ),
                }
                for m in input_dict["messages"]
            ]

            response = await asyncio.to_thread(
                self.client.chat.completions.create, **input_dict
            )
            return LLMResponse(
                raw_response=response,
                parsed_content=response.choices[0].message.content,
            )
        except Exception as error:
            print(f"An error occurred in GroqClient: {str(error)}")
            raise

    def _process_image_content(self, input_data: AskLLMInput):
        for message in reversed(input_data.messages):
            if message["role"] == "user":
                if isinstance(message["content"], str):
                    message["content"] = [{"type": "text", "text": message["content"]}]
                break

    def _create_image_message_from_url(self, url: str) -> dict:
        # Groq doesn't support image inputs, so we'll return a text message instead
        return {"type": "text", "text": f"Image URL: {url}"}

    def _create_image_message_from_filepath(self, filepath: str) -> dict:
        # Groq doesn't support image inputs, so we'll return a text message instead
        return {"type": "text", "text": f"Image file: {filepath}"}
